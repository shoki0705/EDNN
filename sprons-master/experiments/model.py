import scipy
import scipy.sparse.linalg
import sys
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
import torchdyn.numerics
import datetime
import matplotlib.pyplot as plt
from siren_pytorch import SirenNet
#from linalg import CG, GMRES


def get_activation_from_name(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(nn.functional, name):
        return getattr(nn.functional, name)
    if name == "identity":
        return lambda x: x
    raise NotImplementedError(name)


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_hidden_layer: int, nonlinearity: str):
        super(MLP, self).__init__()
        assert hidden_dim > 0
        self.units = [input_dim] + [hidden_dim] * n_hidden_layer + [output_dim]
        self.n_params = sum([self.units[i] * self.units[i + 1] + self.units[i + 1] for i in range(n_hidden_layer + 1)])
        self.act = get_activation_from_name(nonlinearity)
        self.nonlinearity = nonlinearity

    def segment_params(self, params):
        weights = []
        biases = []
        itr_end = 0
        for i in range(len(self.units) - 1):
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i] * self.units[i + 1]
            weights.append(params[itr_stt:itr_end].view(self.units[i + 1], self.units[i]))
            itr_stt = itr_end
            itr_end = itr_stt + self.units[i + 1]
            biases.append(params[itr_stt:itr_end])
        assert torch.numel(params) == itr_end
        return weights, biases

    def init_params(self, params):
        weights, biases = self.segment_params(params)
        try:
            gain = torch.nn.init.calculate_gain(self.nonlinearity)
        except:
            gain = 1.0
        for w in weights:
            nn.init.xavier_normal_(w, gain=gain)
            # nn.init.kaiming_normal(w, nonlinearity="linear" if self.nonlinearity=="selu" else self.nonlinearity)
            # nn.init.kaiming_normal(w, gain=gain)
            # nn.init.xavier_uniform_(w, gain=gain)
            # nn.init.orthogonal_(w, gain=gain)
        for b in biases:
            nn.init.zeros_(b)

    def get_init_params(self):
        theta = torch.zeros(self.n_params)
        self.init_params(theta)
        return theta

    def forward(self, x, params):
        weights, biases = self.segment_params(params)
        with_act = [True] * (len(self.units) - 1) + [False]
        for w, b, a in zip(weights, biases, with_act):
            x = nn.functional.linear(x, w, b)
            if a:
                x = self.act(x)
        return x


class EDNN(nn.Module):
    def __init__(
        self,
        x_range,    # 空間の範囲
        space_dim: int,   # 空間の次元
        state_dim: int,  # 状態の次元
        hidden_dim: int,    # 隠れ層の次元
        n_hidden_layer: int,    # 隠れ層の数
        nonlinearity: str = "tanh",     # 活性化関数
        sinusoidal: int = 0,    # サイン波の数
        is_periodic_boundary: bool = True,  # 周期境界条件
        is_zero_boundary: bool = False, # 境界条件
        space_normalization: bool = True,   # 空間の正規化
    ):
        super(EDNN, self).__init__()
        assert not (is_periodic_boundary and is_zero_boundary)
        assert not is_periodic_boundary or space_normalization  # is_periodic_boundary -> space_normalization

        # x_range: [x1,x2,...] x [min, max]
        self.x_range = torch.from_numpy(x_range)
        self.space_normalization = space_normalization
        self.sinusoidal = sinusoidal
        self.is_periodic_boundary = is_periodic_boundary
        if sinusoidal == 0 and is_periodic_boundary:
            self.sinusoidal = 1
        self.is_zero_boundary = is_zero_boundary

        input_dim = space_dim * ((2 * self.sinusoidal) if self.sinusoidal > 0 else 0)
        self.mlp = MLP(input_dim=input_dim, output_dim=state_dim, hidden_dim=hidden_dim, n_hidden_layer=n_hidden_layer, nonlinearity=nonlinearity)

    def forward(self, x, params):
        # x: batch x [x1,x2,...]
        x_norm = x
        x_range = self.x_range.to(device=x.device, dtype=x.dtype)
        if self.space_normalization:
            x_norm = 2 * (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0]) - 1  # x_norm \in [-1,1]
            x_range = 2 * (x_range - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0]) - 1  # [-1,1]
            if self.sinusoidal>0:
                # x_norm: batch x [sin x1 , sin x2, ..., cos x1, cos x2, ...]
                state_list=[]
                for k in range(1,self.sinusoidal+1):
                    sin_2k_x_norm = torch.sin(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))
                    cos_2k_x_norm = torch.cos(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))
                    state_list.append(sin_2k_x_norm)
                    state_list.append(cos_2k_x_norm)
                x_norm = torch.cat(state_list, dim=-1)
                x_range = x_range.tile(2 * self.sinusoidal).view(len(x_range), 2, 2 * self.sinusoidal).transpose(0, 1).reshape(len(x_range) * 2 * self.sinusoidal, 2)

        u = self.mlp(x_norm, params)

        if self.is_zero_boundary:
            x_pos = (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0])
            u = u * 4 * x_pos * (1 - x_pos)  # u=0 at boundary

        return u

    def u_t(self, x):
        pass

    def get_random_sampling_points(self, N):
        x = torch.rand(N, len(self.x_range))
        x = x * (self.x_range[:, 1] - self.x_range[:, 0]) + self.x_range[:, 0]
        return x

    def get_init_params(self):
        return self.mlp.get_init_params()


class EDNNTrainer(nn.Module):

    def __init__(self, ednn, log_freq=100, logger=print):
        super(EDNNTrainer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.get_default_dtype()

        self.ednn = ednn.to(device=self.device)
        self.log_freq = log_freq
        self.nfe = 0
        self.logger = logger
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initialized: The number of parameters is ", self.ednn.mlp.n_params)



    def learn_initial_condition(self, x0, u0, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC...")
        assert reg >= 0.0
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)

        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        params = nn.Parameter(params)

        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
        
        # 損失の履歴を保存するリスト
        loss_history = []

        with torch.set_grad_enabled(True):
            def closure():
                nonlocal itr
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
                if reg > 0.0:
                    loss = loss + reg * params.__pow__(2).sum()
                if itr % self.log_freq == 0:
                    self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: {itr:*>7}/{max_itr}, Loss: {loss.item():.6e}")
                loss.backward()
                itr += 1
                
                # 損失を履歴に追加
                loss_history.append(loss.item())
                return loss.item()

            if optim.startswith("adam"):
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by Adam...")
                while itr < max_itr:
                    loss = closure()
                    optimizer.step()
                    if loss < atol:
                        break
            elif optim.startswith("lbfgs"):
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by Adam first...")
                while itr < max_itr // 10:
                    loss = closure()
                    optimizer.step()
                self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by LBFGS...")
                optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=max_itr - max_itr // 10, tolerance_grad=atol, tolerance_change=0, history_size=100)
                optimizer.step(closure)
            else:
                raise NotImplementedError(self.optim)

        loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC finished, Loss: {loss.item():.6e}")

        # 損失履歴をプロットして保存
        self.plot_loss_history(loss_history)

        return params.data

    def plot_loss_history(self, loss_history):
        """ 損失履歴をプロットして保存するメソッド """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.title('Loss During Initial Condition Learning')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')  # 損失のスケールを対数にすることも可能
        plt.legend()
        plt.grid()
        plt.savefig(f'loss_history.png')  # プロットをファイルに保存
        plt.close()  # プロットを閉じる

    def get_time_derivative(self, equation, params, method, x, reg):
        params = params.flatten()  # remove batch axis added by torchdiffeq

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            # dim(u) x batch
            u_t = equation(self.ednn(x, params), x).T

        params_to_u = lambda params: self.ednn(x, params)
        # dim(u) x batch x dim(params)
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
            # J = J / self.ednn.mlp.n_params
            M = torch.einsum("dki,dkj->dij", (J, J))
            a = torch.einsum("dki,dk->di", (J, u_t))
            if method == "inversion":
                assert reg == 0.0
                # numerically unstable
                M_inv = torch.linalg.inv(M)
                deriv = torch.einsum("dji,di->dj", (M_inv, a))

                # 1. 逆行列と元の行列の積を表示
                print("逆行列と元の行列の積 (M @ M_inv):")
                print(M @ M_inv)
                
                # 2. 行列の条件数を表示
                condition_number = torch.linalg.cond(M)
                print(f"行列の条件数（最大固有値と最小固有値の比）: {condition_number}")

            elif method == "optimization":
                if reg > 0:
                    a = torch.cat([a, v], dim=1)
                    M = torch.cat([M, K], dim=1)
                    ret = torch.linalg.lstsq(M, a)
                    deriv = ret.solution
                else:
                    deriv = torch.linalg.solve(M, a)
                    print(deriv)
                    
            elif method == "CG":
                deriv, info = CG(M,a)
                
            elif method == "GMRES":
                deriv, info = GMRES(M,a)
                
                
            else:
                raise NotImplementedError(method)

        return deriv

    def forward(self, t, state):
        if self.nfe % self.log_freq == 0 or True:
            assert t.numel() == 1
            self.logger(f"[{datetime.datetime.now()}] EDNN, Integration: time={t.item():.6e}, nfe={self.nfe}, state mean={state.mean()}, var={state.var()}")
        self.nfe += 1
        equation = self.integration_params["equation"]
        method = self.integration_params["method"]
        x_eval = self.integration_params["x_eval"]
        n_eval = self.integration_params["n_eval"]
        reg = self.integration_params["reg"]

        assert x_eval is None or n_eval == 0
        if x_eval is None:
            x_eval = self.ednn.get_random_sampling_points(n_eval).to(device=self.device, dtype=self.dtype)

        deriv = self.get_time_derivative(equation, state, method=method, x=x_eval, reg=reg)
        return deriv

    def integrate(self, params, equation, method, solver, t_eval, x_eval=None, n_eval: int = 0, reg: float = 0.0, atol: float = 1e-3, rtol: float = 1e-3):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Integration")
        # assert method in ["optimization", "inversion", "collocation", "gelsd"]
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

        # t_eval x batch(=1) x params
        if solver.startswith("dyn"):
            params_evolved = torchdyn.numerics.odeint(self, params.reshape(1, -1), t_eval, solver=solver[3:], atol=atol, rtol=rtol, verbose=True)[1]
        else:
            params_evolved = torchdiffeq.odeint(self, params.reshape(1, -1), t_eval, rtol=rtol, atol=atol, method=solver)
        assert isinstance(params_evolved, torch.Tensor)
        params_evolved = params_evolved.squeeze(1)  # remove batch dim

        # t_eval x x_eval x dim(u)
        u_evolved = torch.stack([self.ednn(x_eval, p) for p in params_evolved], dim=0)
        self.nfe = 0

        return u_evolved, params_evolved

    def solver_replacer(self, solver):
        if solver == "rk23":
            return "bosh3"
        if solver == "rk45":
            return "dopri5"
        return solver
