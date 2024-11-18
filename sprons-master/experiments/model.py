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
from siren_pytorch import SirenNet, Sine
#from linalg import CG, GMRES


#   指定された名前の活性化関数を取得
def get_activation_from_name(name):
    if name == "sine":
        return Sine(1.0)
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(nn.functional, name):
        return getattr(nn.functional, name)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Activation function '{name}' is not implemented in torch or torch.nn.functional")


#   多層パーセプトロン
class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_hidden_layer: int, nonlinearity: str):
        super(MLP, self).__init__()
        assert hidden_dim > 0   
        self.units = [input_dim] + [hidden_dim] * n_hidden_layer + [output_dim] # [input, hidden, ..., hidden, output]
        self.n_params = sum([self.units[i] * self.units[i + 1] + self.units[i + 1] for i in range(n_hidden_layer + 1)]) # number of parameters
        self.act = get_activation_from_name(nonlinearity)   # activation function
        self.nonlinearity = nonlinearity    # activation function name

    #   パラメータをweightsとbiasesに分割
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


    #   パラメータの初期化
    def init_params(self, params):
        #   パラメータをweightsとbiasesに分割
        weights, biases = self.segment_params(params)
        #   活性化関数に応じた初期化
        try:
            gain = torch.nn.init.calculate_gain(self.nonlinearity)
        #   例外処理
        except:
            gain = 1.0
        #   重みの初期化
        for w in weights:
            nn.init.xavier_normal_(w, gain=gain)
            # nn.init.kaiming_normal(w, nonlinearity="linear" if self.nonlinearity=="selu" else self.nonlinearity)
            # nn.init.kaiming_normal(w, gain=gain)
            # nn.init.xavier_uniform_(w, gain=gain)
            # nn.init.orthogonal_(w, gain=gain)
        #   バイアスの初期化
        for b in biases:
            nn.init.zeros_(b)

    #   初期パラメータの取得
    def get_init_params(self):
        theta = torch.zeros(self.n_params)  # パラメータ数のゼロテンソル
        self.init_params(theta) # パラメータの初期化
        return theta

    #  順伝播
    def forward(self, x, params):
        weights, biases = self.segment_params(params)   # パラメータをweightsとbiasesに分割
        with_act = [True] * (len(self.units) - 1) + [False] # 活性化関数の有無
        #  順伝播
        for w, b, a in zip(weights, biases, with_act):
            x = nn.functional.linear(x, w, b)   # 線形変換
            # 活性化関数があれば適用
            if a:   
                x = self.act(x)
        return x


class EDNN(nn.Module):
    def __init__(
        self,
        x_range,    # xの範囲
        space_dim: int,   # 空間次元
        state_dim: int,  # 状態次元
        hidden_dim: int,    # 隠れ層の次元
        n_hidden_layer: int,    # 隠れ層の数
        nonlinearity: str = "tanh",   # 活性化関数
        sinusoidal: int = 0,    # サイン波埋め込み
        is_periodic_boundary: bool = True,  # 周期境界条件
        is_zero_boundary: bool = False, # ゼロ境界条件
        space_normalization: bool = True,   # 空間の正規化
    ):
        super(EDNN, self).__init__()    # 親クラスのコンストラクタ
        assert not (is_periodic_boundary and is_zero_boundary)  # 周期境界条件とゼロ境界条件は同時に指定できない
        assert not is_periodic_boundary or space_normalization  # 周期境界条件 -> 空間の正規化

        # x_range: [x1,x2,...] x [min, max]
        self.x_range = torch.from_numpy(x_range)    # xの範囲
        self.space_normalization = space_normalization  # 空間の正規化
        self.sinusoidal = sinusoidal    # サイン波埋め込み
        self.is_periodic_boundary = is_periodic_boundary    # 周期境界条件
        
        # サイン波埋め込みがなく、周期境界条件が指定されている場合
        if sinusoidal == 0 and is_periodic_boundary:    
            self.sinusoidal = 1   # サイン波埋め込みを1に設定
        self.is_zero_boundary = is_zero_boundary    # ゼロ境界条件

        # 入力次元=空間次元 x (2 * サイン波埋め込み)
        input_dim = space_dim * ((2 * self.sinusoidal) if self.sinusoidal > 0 else 0)
        
        # MLPのインスタンス化
        self.mlp = MLP(input_dim=input_dim, output_dim=state_dim, hidden_dim=hidden_dim, n_hidden_layer=n_hidden_layer, nonlinearity=nonlinearity)

    def forward(self, x, params):
        # x: batch x [x1,x2,...]
        x_norm = x  # xの正規化
        x_range = self.x_range.to(device=x.device, dtype=x.dtype)   # xの範囲
        # 空間の正規化
        if self.space_normalization:
            x_norm = 2 * (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0]) - 1  # x_normを正規化
            x_range = 2 * (x_range - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0]) - 1  # x_rangeを正規化
            
            # サイン波埋め込み処理
            if self.sinusoidal>0:
                # x_norm: batch x [sin x1 , sin x2, ..., cos x1, cos x2, ...]
                state_list=[]
                # サイン波埋め込み
                for k in range(1,self.sinusoidal+1):
                    sin_2k_x_norm = torch.sin(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))   # sin(2^k x) / 2^k
                    cos_2k_x_norm = torch.cos(torch.pi * 2**(k-1)*x_norm)*2**(-(k-1))   # cos(2^k x) / 2^k
                    # state_listに追加
                    state_list.append(sin_2k_x_norm)
                    state_list.append(cos_2k_x_norm)
                x_norm = torch.cat(state_list, dim=-1)  # state_listを結合
                # x_rangeを拡張
                x_range = x_range.tile(2 * self.sinusoidal).view(len(x_range), 2, 2 * self.sinusoidal).transpose(0, 1).reshape(len(x_range) * 2 * self.sinusoidal, 2)

        # MLPにx_normとparamsを入力し、出力uを取得
        u = self.mlp(x_norm, params)

        # ゼロ境界条件の場合
        if self.is_zero_boundary:
            x_pos = (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0])
            u = u * 4 * x_pos * (1 - x_pos)  # u=0 at boundary

        return u

    def u_t(self, x):
        pass

    #  ランダムサンプリング点の取得
    def get_random_sampling_points(self, N):
        x = torch.rand(N, len(self.x_range))    # N x [0,1]の乱数
        x = x * (self.x_range[:, 1] - self.x_range[:, 0]) + self.x_range[:, 0]  # x_rangeに変換
        return x

    #  初期パラメータの取得
    def get_init_params(self):
        return self.mlp.get_init_params()

# モデルの訓練
class EDNNTrainer(nn.Module):

    #   コンストラクタ
    def __init__(self, ednn, log_freq=100, logger=print):
        super(EDNNTrainer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.get_default_dtype()  # data type=float64

        self.ednn = ednn.to(device=self.device)   # モデルをGPUに転送
        self.log_freq = log_freq    # ログの頻度
        self.nfe = 0    # 評価回数
        self.logger = logger    # ログ出力
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initialized: The number of parameters is ", self.ednn.mlp.n_params)


    #   初期条件の学習
    def learn_initial_condition(self, x0, u0, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC...")
        assert reg >= 0.0
        
        # x0, u0をGPUに転送
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)

        # パラメータの初期化
        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        params = nn.Parameter(params)

        # 最適化手法の設定
        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
        
        # 損失の履歴を保存するリスト
        loss_history = []

        # 最適化
        with torch.set_grad_enabled(True):
            def closure():
                nonlocal itr
                optimizer.zero_grad()   # 勾配の初期化
                loss = nn.functional.mse_loss(self.ednn(x0, params), u0)    # 損失の計算
                if reg > 0.0:
                    loss = loss + reg * params.__pow__(2).sum() # 正則化項の追加
                # ログの出力
                if itr % self.log_freq == 0:
                    self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: {itr:*>7}/{max_itr}, Loss: {loss.item():.6e}")
                loss.backward() # 逆伝播
                itr += 1
                
                # 損失を履歴に追加
                loss_history.append(loss.item())
                return loss.item()

            # 最適化手法の選択
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

    #   時間微分の計算
    def get_time_derivative(self, equation, params, method, x, reg):
        params = params.flatten()  # remove batch axis added by torchdiffeq

        #  時間微分
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            # dim(u) x batch
            u_t = equation(self.ednn(x, params), x).T

        params_to_u = lambda params: self.ednn(x, params)
        # dim(u) x batch x dim(params)
        J_T = torch.autograd.functional.jacobian(params_to_u, params)   # ヤコビ行列の転置
        assert isinstance(J_T, torch.Tensor)    # ヤコビ行列がテンソルであることを確認
        J = J_T.transpose(0, 1) # ヤコビ行列
        # u_tとJの正則化
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

    #   順伝播
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
