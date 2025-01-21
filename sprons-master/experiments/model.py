import sys
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
import torchdyn.numerics
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import linops as lo
import math
from dataclasses import dataclass
from linear_operator.utils.linear_cg import linear_cg
from scipy.sparse.linalg import gmres, qmr
from cola.linalg.inverse.gmres import gmres
from cola.linalg.inverse.cg import cg
from cola.ops import Dense, Sum
from cola.linalg.preconditioning.preconditioners import NystromPrecond
from cola.linalg.tbd.svrg import solve_svrg_symmetric
from deq.lib.solvers import broyden

#-----------------------------------------------------------------
#   指定された名前の活性化関数を取得
def get_activation_from_name(name):
    if name == "sin":
        # 単純に torch.sin を返す（必要に応じてスケーリングなどを加える）
        return torch.sin
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(nn.functional, name):
        return getattr(nn.functional, name)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Activation function '{name}' is not implemented in torch or torch.nn.functional")


#-----------------------------------------------------------------
#   多層パーセプトロン
class MLP(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int, nonlinearity: str):
        super(MLP, self).__init__()
        assert dim_hidden > 0
        self.units = [dim_in] + [dim_hidden] * num_layers + [dim_out]  # [input, hidden, ..., hidden, output]
        self.n_params = sum([self.units[i] * self.units[i + 1] + self.units[i + 1] for i in range(num_layers + 1)])  # number of parameters
        self.act = get_activation_from_name(nonlinearity)   # activation function
        self.nonlinearity = nonlinearity    # activation function name
        self.num_layers = num_layers    # number of layers
        self.dim_in = dim_in    # input dimension

    # パラメータをweightsとbiasesに分割
    def segment_params(self, params):
        weights = []    # weightsを格納
        biases = []    # biasesを格納
        itr_end = 0
        for i in range(len(self.units) - 1):
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i] * self.units[i + 1]
            weights.append(params[itr_stt:itr_end].view(self.units[i + 1], self.units[i]))  # weight
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i + 1]
            biases.append(params[itr_stt:itr_end])  # bias
        assert torch.numel(params) == itr_end   # check
        return weights, biases

    # weightsとbiasesを結合
    def concat_params(self, weights, biases):
        params_list = []
        for w, b in zip(weights, biases):
            # weightをフラット化
            params_list.append(w.view(-1))
            # biasもフラット化
            params_list.append(b.view(-1))
        # 全てのパラメータをまとめて1つのテンソルに結合
        params = torch.cat(params_list)
        return params

    # パラメータの初期化
    def init_params(self, params):
        # パラメータをweightsとbiasesに分割
        weights, biases = self.segment_params(params)
        # 活性化関数に応じた初期化
        try:
            gain = torch.nn.init.calculate_gain(self.nonlinearity)
        except:
            gain = 1.0
        # 重みの初期化
        for w in weights:
            nn.init.xavier_normal_(w, gain=gain)
        # バイアスの初期化
        for b in biases:
            nn.init.zeros_(b)

    # 初期パラメータの取得
    def get_init_params(self):
        theta = torch.zeros(self.n_params)
        self.init_params(theta)
        return theta

    # 順伝播
    def forward(self, x, params, hidden=False):
        weights, biases = self.segment_params(params)  # パラメータをweightsとbiasesに分割
        with_act = [True] * (len(weights) - 1) + [False]  # 活性化関数の有無

        # 順伝播
        if self.nonlinearity == "sin":
            for i, (w, b, a) in enumerate(zip(weights, biases, with_act)):
                x = nn.functional.linear(x, w, b)  # 線形変換
                if a:
                    # 入力層では異なるスケールを用いる例（必要に応じて変更）
                    omega = 30 if i == 0 else 1  
                    x = torch.sin(omega * x)
                # 最終層への入力
                if i == len(weights) - 2 and hidden:
                    return x
        else:
            for i, (w, b, a) in enumerate(zip(weights, biases, with_act)):
                x = nn.functional.linear(x, w, b)
                if a:
                    x = self.act(x)
                if i == len(weights) - 2 and hidden:
                    return x
        return x

#-----------------------------------------------------------------
# 以下、EDNN, EDNNTrainer 等の定義は省略せずに記載（質問文のコードをそのまま利用）

class EDNN(nn.Module):
    def __init__(
        self,
        x_range,    # xの範囲
        space_dim: int,   # 空間次元
        state_dim: int,  # 状態次元
        dim_hidden: int,    # 隠れ層の次元
        num_layers: int,    # 隠れ層の数
        nonlinearity: str = "sin",   # 活性化関数
        sinusoidal: int = 0,    # サイン波埋め込み
        is_periodic_boundary: bool = True,  # 周期境界条件
        is_zero_boundary: bool = False, # ゼロ境界条件
        space_normalization: bool = True,   # 空間の正規化
    ):
        super(EDNN, self).__init__()
        assert not (is_periodic_boundary and is_zero_boundary)
        assert not is_periodic_boundary or space_normalization

        self.x_range = torch.from_numpy(x_range)
        self.space_dim = space_dim
        self.space_normalization = space_normalization
        self.sinusoidal = sinusoidal
        self.is_periodic_boundary = is_periodic_boundary

        if sinusoidal == 0 and is_periodic_boundary:
            self.sinusoidal = 1
        self.is_zero_boundary = is_zero_boundary

        dim_in = space_dim * ((2 * self.sinusoidal) if self.sinusoidal > 0 else 0)
        self.mlp = MLP(dim_in=dim_in, dim_out=state_dim, dim_hidden=dim_hidden, num_layers=num_layers, nonlinearity=nonlinearity)

    def forward(self, x, params, hidden=False):
        x_norm = x
        x_range = self.x_range.to(device=x.device, dtype=x.dtype)
        if self.space_normalization:
            x_norm = 2 * (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0]) - 1
            x_range = 2 * (x_range - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0]) - 1
            if self.sinusoidal > 0:
                state_list = []
                for k in range(1, self.sinusoidal + 1):
                    sin_2k_x_norm = 1.5 * torch.sin(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))
                    cos_2k_x_norm = 1.5 * torch.cos(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))
                    state_list.append(sin_2k_x_norm)
                    state_list.append(cos_2k_x_norm)
                x_norm = torch.cat(state_list, dim=-1)
                x_range = x_range.tile(2 * self.sinusoidal).view(len(x_range), 2, 2 * self.sinusoidal).transpose(0, 1).reshape(len(x_range) * 2 * self.sinusoidal, 2)
        u = self.mlp(x_norm, params, hidden)
        if self.is_zero_boundary and (not hidden):
            x_pos = (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0])
            u = u * 4 * x_pos * (1 - x_pos)
        return u

    def get_random_sampling_points(self, N):
        x = torch.rand(N, len(self.x_range))
        x = x * (self.x_range[:, 1] - self.x_range[:, 0]) + self.x_range[:, 0]
        return x

    def get_init_params(self):
        return self.mlp.get_init_params()

#-----------------------------------------------------------------
# モデルの訓練
class EDNNTrainer(nn.Module):

    def __init__(self, ednn, log_freq=100, restart_times=10, logger=print):
        super(EDNNTrainer, self).__init__()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.get_default_dtype()
        self.ednn = ednn.to(device=self.device)
        self.log_freq = log_freq
        self.restart_times = restart_times
        self.restart_flag = [True]*restart_times
        self.nfe = 0
        self.logger = logger
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initialized: The number of parameters is ", self.ednn.mlp.n_params)

    def learn_initial_condition(self, x0, u0, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000, batch_size: int = 1000):
        self.x0 = x0
        self.batch_size = batch_size
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC...")
        assert reg >= 0.0

        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)
        dataset = torch.utils.data.TensorDataset(x0, u0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        self.params_shape = params.shape
        params = nn.Parameter(params)
        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
        loss_history = []
        with tqdm(total=max_itr, desc="Training Progress", unit="step") as pbar:
            with torch.set_grad_enabled(True):
                for epoch in range(max_itr // len(dataloader)):
                    for batch_x, batch_u in dataloader:
                        optimizer.zero_grad()
                        loss = nn.functional.mse_loss(self.ednn(batch_x, params, hidden=False), batch_u)
                        if reg > 0.0:
                            loss = loss + reg * params.__pow__(2).sum()
                        loss.backward()
                        optimizer.step()
                        itr += 1
                        loss_history.append(loss.item())
                        pbar.set_postfix(loss=f"{loss.item():.6e}")
                        pbar.update(1)
                        if loss.item() < atol or itr >= max_itr:
                            break
                    if loss.item() < atol or itr >= max_itr:
                        break
        loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC finished, Loss: {loss.item():.6e}")
        self.plot_loss_history(loss_history, optim, max_itr)
        return params.data

    def plot_loss_history(self, loss_history, optim : str = "Adam", max_itr = "10000"):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.title('Loss During Initial Condition Learning')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(f'loss_history_{optim}_itr{max_itr}.png')
        plt.close()

    def solve_head(self, phi_theta, u0, reg: float = 1e-5):
        n_eval = phi_theta.shape[0]
        n_features = phi_theta.shape[1]
        if u0.dim() == 1:
            u0 = u0.unsqueeze(1)
        Phi = torch.cat(
            [phi_theta, torch.ones(n_eval, 1, device=self.device, dtype=self.dtype)], dim=1
        )
        A = Phi.T @ Phi
        if reg > 0:
            reg_matrix = reg * torch.eye(A.size(0), device=self.device, dtype=self.dtype)
            A += reg_matrix
        b = Phi.T @ u0
        w_b = torch.linalg.solve(A, b)
        w = w_b[:-1].detach()
        b = w_b[-1].detach()
        return w, b

    def head_tuning(self, params, x0_in, u0_in, initial_equation=None, hreg: float = 1e-5, N=100000):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Head tuning started...")
        if initial_equation is not None:
            x0 = self.ednn.get_random_sampling_points(N)
            x0 = x0.cpu().numpy()
            u0 = initial_equation(x0)
            x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
            u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)
        else:
            x0 = torch.from_numpy(x0_in).to(device=self.device, dtype=self.dtype)
            u0 = torch.from_numpy(u0_in).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            phi_theta = self.ednn(x0, params, hidden=True)
        with torch.no_grad():
            predicted_u0 = self.ednn(x0, params)
        initial_loss = nn.functional.mse_loss(predicted_u0, u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initial Loss: {initial_loss.item():.6e}")
        w, b = self.solve_head(phi_theta, u0, hreg)
        weights, biases = self.ednn.mlp.segment_params(params)
        weights[-1].copy_(w.view_as(weights[-1]))
        biases[-1].copy_(b)
        params = self.ednn.mlp.concat_params(weights, biases)
        params = nn.Parameter(params)
        with torch.no_grad():
            predicted_u0 = self.ednn(x0, params)
        final_loss = nn.functional.mse_loss(predicted_u0, u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Final Loss: {final_loss.item():.6e}")
        self.logger(f"[{datetime.datetime.now()}] EDNN, Head tuning finished.")
        return params
    
    def I_1(self, u, x):
        return u.sum(dim=-1)

    def get_nabla_I(self, u_q, I):
        x_samples = self.ednn.get_random_sampling_points(2000).to(device=self.device, dtype=self.dtype)
        I_values = I(u_q, x_samples)
        
        return I_values
        

    def get_b(self, deriv, u_q):
        b = u_q * deriv
        return b
        

    def get_lambda(self, deriv, u_q):
        nabla_I= self.get_nabla_I(u_q, )
        
        return nabla_I

    # 時間微分の計算
    def get_time_derivative(self, equation, params, method, x, reg, conserved=True):
        params = nn.Parameter(params.flatten(), requires_grad=True)  # remove batch axis if any
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            u_t = equation(self.ednn(x, params), x).T
        params_to_u = lambda params: self.ednn(x, params)
        J_T = torch.autograd.functional.jacobian(params_to_u, params)
        assert isinstance(J_T, torch.Tensor)
        J = J_T.transpose(0, 1)
        if reg > 0:
            u_dim = u_t.shape[0]
            n_param = params.shape[0]
            v = torch.zeros([u_dim, n_param], device=u_t.device, dtype=u_t.dtype)
            K = np.sqrt(reg / u_dim) * torch.eye(n_param, device=u_t.device, dtype=u_t.dtype).expand(u_dim, -1, -1)
        if method == "collocation":
            if reg > 0:
                u_t = torch.cat([u_t, v], dim=1)
                J = torch.cat([J, K], dim=1)
            ret = torch.linalg.lstsq(J, u_t)
            deriv = ret.solution
        elif method == "gelsd":
            if reg > 0:
                u_t = torch.cat([u_t, v], dim=1)
                J = torch.cat([J, K], dim=1)
            ret = torch.linalg.lstsq(J.cpu(), u_t.cpu(), driver="gelsd")
            deriv = ret.solution.to(device=J.device)
        else:
            M = torch.einsum("dki,dkj->dij", (J, J))
            a = torch.einsum("dki,dk->di", (J, u_t))
            if method == "inversion":
                assert reg == 0.0
                M_inv = torch.linalg.inv(M)
                deriv = torch.einsum("dji,di->dj", (M_inv, a))
            elif method == "optimization":
                if reg > 0:
                    a = torch.cat([a, v], dim=1)
                    M = torch.cat([M, K], dim=1)
                    ret = torch.linalg.lstsq(M, a)
                    deriv = ret.solution
                else:
                    deriv = torch.linalg.solve(M, a)
            elif method == "NysPCG":
                M = M.squeeze(0)
                a = a.squeeze(0)
                M_ = Dense(M)
                Nys = NystromPrecond(M_, rank=100, mu=1e-6)
                P = Nys @ torch.eye(M_.shape[0], M.shape[0], dtype=self.dtype, device=self.device)
                deriv, info = cg(A=M_, rhs=a, P=P, max_iters=1000, tol=1e-6)
                deriv = deriv.unsqueeze(0)
                M = M.unsqueeze(0)
                a = a.unsqueeze(0)
            elif method == "CG":
                M = M.squeeze(0)
                a = a.squeeze(0)
                M = Dense(M)
                deriv, info = cg(M, a, max_iters=1000, tol=1e-6)
                deriv = deriv.unsqueeze(0)
            elif method == "GMRES":
                M = M.squeeze(0)
                a = a.squeeze(0)
                M = Dense(M)
                deriv, info = gmres(M, a, max_iters=1000, tol=1e-6)
                deriv = deriv.unsqueeze(0)
            elif method == "QMR":
                M = M.squeeze(0).cpu().numpy()
                a = a.squeeze(0).cpu().numpy()
                deriv, info = qmr(M, a, tol=1e-6)
                deriv = torch.tensor(deriv, dtype=self.dtype, device=self.device)
                deriv = deriv.unsqueeze(0)
            elif method == "SVRG":
                M = M.squeeze(0)
                a = a.squeeze(0)
                M = Dense(M)
                Nys = NystromPrecond(M, 200)
                components = [Dense(J[b_idx].T @ J[b_idx]) for b_idx in range(J.shape[0])]
                M_sum = Sum(*components)
                P = Nys @ torch.eye(M.shape[0], M.shape[0], dtype=self.dtype, device=self.device)
                deriv, _ = solve_svrg_symmetric(M_sum, a, tol=1e-6, P=P)
            elif method == "broyden":
                M = M.squeeze(0)
                a = a.squeeze(0)
                x0 = torch.zeros([M.shape[0], 1], device=u_t.device, dtype=u_t.dtype)
                x0 = x0.unsqueeze(0)
                f = lambda x: M @ x - a
                threshold = 50
                eps = 1e-6
                stop_mode = "rel"
                ls = False
                deriv = broyden(f, x0, threshold=threshold, eps=eps, stop_mode=stop_mode, ls=ls)
                deriv = deriv.unsqueeze(1)
            else:
                raise NotImplementedError(method)
            
        if conserved:
            # u_current は self.ednn(x, params) で計算
            u_current = params_to_u(params)
            u_q = torch.autograd.grad(self.ednn(x, params), params, grad_outputs=torch.ones_like(self.ednn(x, params)), create_graph=True)
            print("u_q", u_q.shape)
            print("deriv", deriv.shape)
            
            lambda_ = self.get_lambda(deriv, u_q)
            a = a - lambda_ * u_current
            
            M = M.squeeze(0)
            a = a.squeeze(0)
            M_ = Dense(M)
            Nys = NystromPrecond(M_, rank=100, mu=1e-6)
            P = Nys @ torch.eye(M_.shape[0], M.shape[0], dtype=self.dtype, device=self.device)
            deriv, info = cg(A=M_, rhs=a, P=P, max_iters=1000, tol=1e-6)
            deriv = deriv.unsqueeze(0)
            
            
        return deriv

    def retraining(self, params_pre, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000, batch_size: int = 1000):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training model...")
        assert reg >= 0.0
        x0 = self.x0
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            u0 = self.ednn(x0, params_pre, hidden=False)
        u0 = u0.to(device=self.device, dtype=self.dtype)
        dataset = torch.utils.data.TensorDataset(x0, u0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        params = nn.Parameter(params)
        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
        with tqdm(total=max_itr, desc="Re-training Progress", unit="step") as pbar:
            with torch.set_grad_enabled(True):
                for epoch in range(max_itr // len(dataloader)):
                    for batch_x, batch_u in dataloader:
                        optimizer.zero_grad()
                        loss = nn.functional.mse_loss(self.ednn(batch_x, params, hidden=False), batch_u)
                        if reg > 0.0:
                            loss = loss + reg * params.__pow__(2).sum()
                        loss.backward()
                        optimizer.step()
                        itr += 1
                        pbar.set_postfix(loss=f"{loss.item():.6e}")
                        pbar.update(1)
                        if loss.item() < atol or itr >= max_itr:
                            break
                    if loss.item() < atol or itr >= max_itr:
                        break
        loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training: IC finished, Loss: {loss.item():.6e}")
        torch.cuda.empty_cache()
        return params.data

    def forward(self, t, state):
        if self.nfe % self.log_freq == 0 or True:
            assert t.numel() == 1
            self.logger(f"[{datetime.datetime.now()}] EDNN, Integration: time={t.item():.6e}, nfe={self.nfe}, state mean={state.mean()}, var={state.var()}")
        self.nfe += 1
        restart_idx = self.restart_flag.index(True)
        if t.item() > float((restart_idx+1)/self.restart_times)  and any(self.restart_flag):
            self.logger(f"[{datetime.datetime.now()}] EDNN, Restart condition")
            params_pre = state.reshape(self.params_shape)
            params_new = self.retraining(
                params_pre, 
                reg=0.0, 
                optim="adam", 
                lr=1e-3, 
                atol=1e-7, 
                max_itr=10000, 
                batch_size=self.batch_size
            )
            state = params_new.reshape(1, -1)
            self.logger(f"[{datetime.datetime.now()}] EDNN, Restarted: Updated parameters.")
            self.restart_flag[restart_idx] = False
        equation = self.integration_params["equation"]
        method = self.integration_params["method"]
        x_eval = self.integration_params["x_eval"]
        n_eval = self.integration_params["n_eval"]
        reg = self.integration_params["reg"]
        assert x_eval is None or n_eval == 0
        if x_eval is None:
            x_eval = self.ednn.get_random_sampling_points(n_eval).to(device=self.device, dtype=self.dtype)
        deriv = self.get_time_derivative(equation, state, method=method, x=x_eval, reg=reg, conserved=False)
        return deriv

    def integrate(self, params, equation, method, solver, t_eval, x_eval=None, n_eval: int = 0, reg: float = 0.0, atol: float = 1e-3, rtol: float = 1e-3):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Integration")
        assert t_eval[0] == 0
        assert reg >= 0.0
        t_eval = torch.tensor(t_eval, device=self.device, dtype=self.dtype)
        x_eval = torch.tensor(x_eval, device=self.device, dtype=self.dtype)
        solver = self.solver_replacer(solver)
        self.nfe = 0
        if n_eval > 0:
            self.integration_params = dict(equation=equation, method=method, x_eval=None, n_eval=n_eval, reg=reg)
        else:
            self.integration_params = dict(equation=equation, method=method, x_eval=x_eval, n_eval=0, reg=reg)
        if solver.startswith("dyn"):
            params_evolved = torchdyn.numerics.odeint(self, params.reshape(1, -1), t_eval, solver=solver[3:], atol=atol, rtol=rtol, verbose=True)[1]
        else:
            params_evolved = torchdiffeq.odeint(self, params.reshape(1, -1), t_eval, rtol=rtol, atol=atol, method=solver)
        assert isinstance(params_evolved, torch.Tensor)
        params_evolved = params_evolved.squeeze(1)
        u_evolved = torch.stack([self.ednn(x_eval, p) for p in params_evolved], dim=0)
        self.nfe = 0
        return u_evolved, params_evolved

    def solver_replacer(self, solver):
        if solver == "rk23":
            return "bosh3"
        if solver == "rk45":
            return "dopri5"
        return solver
