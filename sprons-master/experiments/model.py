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
from tqdm import tqdm
import linops as lo
import math
from dataclasses import dataclass




#   指定された名前の活性化関数を取得
def get_activation_from_name(name):
    if name == "sin":
        return 
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(nn.functional, name):
        return getattr(nn.functional, name)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Activation function '{name}' is not implemented in torch or torch.nn.functional")


#   多層パーセプトロン
class MLP(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int, nonlinearity: str):
        super(MLP, self).__init__()
        assert dim_hidden > 0   
        self.units = [dim_in] + [dim_hidden] * num_layers + [dim_out] # [input, hidden, ..., hidden, output]
        self.n_params = sum([self.units[i] * self.units[i + 1] + self.units[i + 1] for i in range(num_layers + 1)]) # number of parameters
        self.act = get_activation_from_name(nonlinearity)   # activation function
        self.nonlinearity = nonlinearity    # activation function name
        self.num_layers = num_layers    # number of layers

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

    # 順伝播
    def forward(self, x, params, hidden=False):
        weights, biases = self.segment_params(params)  # パラメータをweightsとbiasesに分割
        with_act = [True] * (len(self.units) - 1) + [False]  # 活性化関数の有無
        
        # 最終層への入力を保持する変数
        final_layer_input = None

        # 順伝播
        if self.nonlinearity == "sin":
            for i, (w, b, a) in enumerate(zip(weights, biases, with_act)):
                x = nn.functional.linear(x, w, b)  # 線形変換
                if a:
                    omega = 30 if i == 0 else 1  # 入力層では w=30、隠れ層では w=1
                    x = torch.sin(omega * x)  # サイン活性化関数を適用
                    
                # 最終層への入力を返す
                if i == len(weights) - 2 and hidden:
                    final_layer_input = x.clone()
                    return final_layer_input
        else:
            for i, (w, b, a) in enumerate(zip(weights, biases, with_act)):
                x = nn.functional.linear(x, w, b)  # 線形変換
                # 活性化関数があれば適用
                if a:
                    x = self.act(x)
                # 最終層への入力を返す
                if i == len(weights) - 2 and hidden:
                    final_layer_input = x.clone()
                    return final_layer_input
        
        # 通常の順伝播の出力を返す
        return x

    

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
        super(EDNN, self).__init__()    # 親コンストラクタ
        assert not (is_periodic_boundary and is_zero_boundary)  # 周期境界条件とゼロ境界条件は同時に指定できない
        assert not is_periodic_boundary or space_normalization  # 周期境界条件 -> 空間の正規化

        # x_range: [x1,x2,...] x [min, max]
        self.x_range = torch.from_numpy(x_range)    # xの範囲
        self.space_dim = space_dim  # 空間次元
        self.space_normalization = space_normalization  # 空間の正規化
        self.sinusoidal = sinusoidal    # サイン波埋め込み
        self.is_periodic_boundary = is_periodic_boundary    # 周期境界条件
        
        # サイン波埋め込みがなく、周期境界条件が指定されている場合
        if sinusoidal == 0 and is_periodic_boundary:    
            self.sinusoidal = 1   # サイン波埋め込みを1に設定
        self.is_zero_boundary = is_zero_boundary    # ゼロ境界条件

        # 入力次元=空間次元 x (2 * サイン波埋め込み)
        dim_in = space_dim * ((2 * self.sinusoidal) if self.sinusoidal > 0 else 0)
        
        # MLPのインスタンス化
        self.mlp = MLP(dim_in=dim_in, dim_out=state_dim, dim_hidden=dim_hidden, num_layers=num_layers, nonlinearity=nonlinearity)

    def forward(self, x, params, hidden=False):
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
        u = self.mlp(x_norm, params, hidden)

        # ゼロ境界条件の場合
        if self.is_zero_boundary:
            x_pos = (x_norm - x_range[None, :, 0]) / (x_range[None, :, 1] - x_range[None, :, 0])
            u = u * 4 * x_pos * (1 - x_pos)  # u=0 at boundary

        return u

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
    def __init__(self, ednn, log_freq=100, restart_fleq=400, logger=print):
        super(EDNNTrainer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.get_default_dtype()  # data type=float64

        self.ednn = ednn.to(device=self.device)   # モデルをGPUに転送
        self.log_freq = log_freq    # ログの頻度
        self.restart_fleq = restart_fleq    # リスタートの頻度
        self.nfe = 0    # 評価回数
        self.logger = logger    # ログ出力
        self.logger(f"[{datetime.datetime.now()}] EDNN, Initialized: The number of parameters is ", self.ednn.mlp.n_params)


    def learn_initial_condition(self, x0, u0, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000):
        self.x0 = x0
        self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC...")
        assert reg >= 0.0
        
        # x0, u0をGPUに転送
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)

        # パラメータの初期化
        params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
        self.params_shape = params.shape
        params = nn.Parameter(params)

        # 最適化手法の設定
        itr = 0
        optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
        
        # 損失の履歴を保存するリスト
        loss_history = []

        # プログレスバーを設定
        with tqdm(total=max_itr, desc="Training Progress", unit="step") as pbar:
            # 最適化
            with torch.set_grad_enabled(True):
                def closure():
                    nonlocal itr
                    optimizer.zero_grad()   # 勾配の初期化
                    loss = nn.functional.mse_loss(self.ednn(x0, params, hidden=False), u0)    # 損失の計算
                    if reg > 0.0:
                        loss = loss + reg * params.__pow__(2).sum()  # 正則化項の追加
                    loss.backward()  # 逆伝播
                    itr += 1
                    loss_history.append(loss.item())  # 損失を履歴に追加

                    # プログレスバーの更新
                    pbar.set_postfix(loss=f"{loss.item():.6e}")  # 損失値を表示
                    pbar.update(1)  # プログレスバーを1ステップ進める
                    return loss.item()

                # 最適化手法の選択
                if optim.startswith("adam"):
                    self.logger(f"[{datetime.datetime.now()}] EDNN, Learning: IC by Adam...")
                    while itr < max_itr:
                        loss = closure()
                        optimizer.step()
                        if loss < atol: # 損失が許容誤差以下になったら終了
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
        self.plot_loss_history(loss_history, optim, max_itr)
        
        return params.data
    
    
    def plot_loss_history(self, loss_history, optim : str = "Adam", max_itr = "10000"):
        """ 損失履歴をプロットして保存するメソッド """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.title('Loss During Initial Condition Learning')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')  # 損失のスケールを対数にすることも可能
        plt.legend()
        plt.grid()
        plt.savefig(f'loss_history_{optim}_itr{max_itr}.png')  # プロットをファイルに保存
        plt.close()  # プロットを閉じる


    def solve_head(self, phi_theta, u0, reg: float = 1e-5):
        """
        最小二乗問題を解き、最終層の重みとバイアスを計算する。

        :param phi_theta: 最終層への入力 (n_eval, feature_dim)
        :param u0: 真の出力データ (n_eval, 1)
        :param reg: 正則化項の係数
        :return: 最終層の重み w とバイアス b
        """
        n_eval = phi_theta.shape[0]
        n_features = phi_theta.shape[1]
        
        # u0 の形状を確認・調整
        if u0.dim() == 1:
            u0 = u0.unsqueeze(1)  # (n_eval,) -> (n_eval, 1)

        # 入力行列 Phi を構築
        Phi = torch.cat(
            [phi_theta, torch.ones(n_eval, 1, device=self.device, dtype=self.dtype)], dim=1
        )  # (n_eval, n_features + 1)


        # 正則化行列を追加
        if reg > 0:
            # 正則化行列 reg_matrix を作成（サイズ: (n_features + 1, n_features + 1)）
            reg_matrix = reg * torch.eye(n_features + 1, device=self.device, dtype=self.dtype)

            # 正則化用の右辺 reg_vector を作成（サイズ: (n_features + 1, 1)）
            reg_vector = torch.zeros(n_features + 1, 1, device=self.device, dtype=self.dtype)

            # Phi に正則化項を適用するために拡張
            Phi = torch.cat([Phi, torch.sqrt(reg_matrix)], dim=0)  # (n_eval + n_features + 1, n_features + 1)
            u0 = torch.cat([u0, reg_vector], dim=0)  # (n_eval + n_features + 1, 1)


        # 条件数を計算
        A = Phi.T @ Phi
        condition_number = torch.linalg.cond(A)
        self.logger(f"係数行列の条件数（最大固有値と最小固有値の比）: {condition_number}")

        # 最小二乗問題を解く
        result = torch.linalg.lstsq(Phi.cpu(), u0.cpu(), driver="gelsd")
        w_b = result.solution.to(device=Phi.device)

        # 重みとバイアスを分割
        w = w_b[:-1]  # 最終層の重み (n_features, 1)
        b = w_b[-1]   # 最終層のバイアス

        return w, b, Phi, w_b



    def head_tuning(self, params, x0, u0, reg: float = 1e-5):
        self.logger(f"[{datetime.datetime.now()}] EDNN, Head tuning started...")

        
        
        
        
        
        # 入力データをテンソルに変換し、デバイスに転送
        x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
        u0 = torch.from_numpy(u0).to(device=self.device, dtype=self.dtype)

        # 中間層の出力を取得（最終層への入力）
        with torch.no_grad():
            phi_theta = self.ednn(x0, params, hidden=True)  # (n_eval, feature_dim)

        
        # 最小二乗問題を解く
        w, b, Phi, w_b = self.solve_head(phi_theta, u0, reg)  # 重みとバイアスを計算

        # パラメータに反映
        weights, biases = self.ednn.mlp.segment_params(params)
        weights[-1].copy_(w.view_as(weights[-1]))  # 重みを更新
        biases[-1].copy_(b)  # バイアスを更新

        # モデルを用いて推定出力を計算
        with torch.no_grad():
            predicted_u0 = self.ednn(x0, params)  # モデルによる推定出力

        # 損失を計算
        final_loss = nn.functional.mse_loss(predicted_u0, u0)  # MSE損失

        self.logger(f"[{datetime.datetime.now()}] EDNN, Head tuning finished. Final Loss: {final_loss.item():.6e}")
        
        return params


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
                    
            elif method == "CG":
                # 共役勾配法 (Conjugate Gradient Method) を用いて最小二乗法 J^T J @ deriv = J^T u_t を解く。
                
                # パラメータ:
                    # J (torch.Tensor): ヤコビ行列 (形状: [dim(u), dim(params), batch])
                    # u_t (torch.Tensor): u_t (形状: [dim(u), batch])
                    # a (torch.Tensor): J^T @ u_t (形状: [dim(params), batch])
                    # M (torch.Tensor): J^T @ J (形状: [dim(params), dim(params), batch])
                    # tol (float): 収束の許容誤差。
                    # max_iter (int): 最大反復回数。

                # 戻り値:
                    # deriv (torch.Tensor): 解ベクトル (形状: [dim(params), batch])。

                # 初期化
                deriv = torch.zeros_like(a)  # 解の初期値
                max_iter = 1000
                tol = 1e-6
                r = a - torch.einsum("dij,dj->di", (M, deriv))  # 残差ベクトル
                p = r.clone()  # 共役方向の初期値
                rs_old = torch.einsum("di,di->d", r, r)  # 残差の内積

                for _ in range(max_iter):
                    # p に対する M の積を計算
                    Mp = torch.einsum("dij,dj->di", (M, p))
                    
                    # アルファ係数の計算
                    alpha = rs_old / torch.einsum("di,di->d", p, Mp)
                    
                    # 解を更新
                    deriv += alpha.unsqueeze(-1) * p

                    # 残差を更新
                    r -= alpha.unsqueeze(-1) * Mp

                    # 残差のノルムが収束条件を満たした場合終了
                    rs_new = torch.einsum("di,di->d", r, r)
                    if torch.sqrt(rs_new).max() < tol:
                        break

                    # ベータ係数の計算
                    beta = rs_new / rs_old
                    p = r + beta.unsqueeze(-1) * p  # 共役方向を更新
                    rs_old = rs_new
            
            elif method == "gpytorchCG":
                from linear_operator.utils.linear_cg import linear_cg

                # ヤコビアンの積を計算するクロージャを定義
                def matmul_closure(v):
                    """
                    M @ v を計算するクロージャ
                    """
                    return torch.einsum("dij,dj->di", (M, v))

                # GPyTorchのlinear_cgを用いて解を計算
                max_iter = 10000
                tol = 1e-6

                deriv = linear_cg(matmul_closure, a, tolerance=tol, max_iter=max_iter)
                
            elif method == "PreCG":
                from linear_operator.utils.linear_cg import linear_cg
                subspace_rank=100
                reg=1e-6

                # ヤコビアンの積を計算するクロージャを定義
                def matmul_closure(v):
                    """
                    M @ v を計算するクロージャ
                    """
                    
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    
                    return torch.einsum("dij,dj->di", (M, v))

                # プリコンディショナーを計算するクロージャ
                def preconditioner_closure(v):
                    # プリコンディショナーをキャッシュ
                    if not hasattr(self, "_precomputed_P") or self._precomputed_P is None:
                        self._precomputed_P = self.compute_nystrom_preconditioner(M, reg)
                    P = self._precomputed_P
                    # 入力 v が P と同じデバイスにあることを確認
                    if v.device != P.A_hat.U.device:
                        v = v.to(P.A_hat.U.device)
                    
                    
                    if v.dim() == 2 and v.size(0) == 1:
                        v = v.squeeze(0)
                    
                    # P_inv(v) を計算
                    U = P.A_hat.U
                    Lambda_hat = P.A_hat.Lambda_hat
                    mu = P.mu

                    # プリコンディショナーを適用 (P^{-1}v)
                    v_projected = U.T @ v  # U^T @ v
                    scaled_projection = (v_projected / (Lambda_hat + mu))  # スケーリング
                    preconditioned_v = U @ scaled_projection + (v - U @ v_projected)

                    preconditioned_v = preconditioned_v.unsqueeze(0)
                    
                    return preconditioned_v


                # GPyTorchのlinear_cgを用いて解を計算
                max_iter = 1000
                tol = 1e-6

                # プリコンディショナー付き共役勾配法
                deriv = linear_cg(matmul_closure, a, tolerance=tol, max_iter=max_iter, preconditioner=preconditioner_closure)

                                            
            else:
                raise NotImplementedError(method)

        return deriv
    
    
    def construct_approximation(
            self,
            A: lo.LinearOperator,
            l_0: int,
            l_max: int,
            power_iter_count: int,
            error_tol, dtype=None):
        n = A.shape[0]
        assert A.shape[0] == A.shape[1]
        device = self.device

        Y = torch.empty((n, 0), dtype=dtype, device=device)
        Omega = torch.empty((n, 0), dtype=dtype, device=device)
        E = torch.inf
        m = l_0
        break_early = False
        U = torch.Tensor([])
        Lambda_hat = torch.Tensor([])
        nu = math.sqrt(n) * torch.finfo().eps

        while E > error_tol:
            Omega_0 = torch.randn((n, m), dtype=dtype, device=device)
            Omega_0, _ = torch.linalg.qr(Omega_0)
            Y_0 = lo.operator_matrix_product(A, Omega_0)
            Omega = torch.hstack([Omega, Omega_0])
            Y = torch.hstack([Y, Y_0])
            Y_nu = Y + nu * Omega
            C = torch.linalg.cholesky(Omega.T @ Y_nu).T
            B = torch.linalg.solve_triangular(C, Y_nu, upper=True, left=False)
            U, Sigma, _ = torch.linalg.svd(B, False)
            Lambda_hat = torch.relu(Sigma * Sigma - nu)

            if break_early:
                break

            E = lo.nystrom_precondition.randomized_power_err_est(A, U, Lambda_hat, power_iter_count, dtype, device)
            m, l_0 = l_0, 2 * l_0
            if l_0 > l_max:
                l_0 = l_0 - m
                m = l_max - l_0
                break_early = True
        #Add threshold to cut off too small values
        return lo.nystrom_precondition._NystromApproximation(U, Lambda_hat)
    
    def compute_nystrom_preconditioner(self, M, reg=1e-6):
        """
        ランダム化Nystromプリコンディショナーを計算する。
        """
        # M がバッチ次元を持っている場合は squeeze しておく
        M = M.squeeze(0)
        
        linear_operator_M = lo.aslinearoperator(M)
        l_0 = 100
        l_max = 300
        power_iter_count = 2  # パワーイテレーションの回数
        error_tol = 1e-6  # 許容誤差
        
        
        M_hat = self.construct_approximation(
            A=linear_operator_M,
            l_0=l_0,
            l_max=l_max,
            power_iter_count=power_iter_count,
            error_tol=error_tol,
            dtype=self.dtype,
        )
        
        mu = torch.tensor(reg, dtype=self.dtype, device=self.device)
        P = lo.nystrom_precondition.NystromPreconditioner(A_hat=M_hat, mu=mu)
        
        return P



    # モデルの再学習
    def retraining(self, params_pre, reg: float = 0.0, optim: str = "adam", lr: float = 1e-3, atol: float = 1e-7, max_itr: int = 1000000):
            self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training model...")
            assert reg >= 0.0
            x0 = self.x0
            x0 = torch.from_numpy(x0).to(device=self.device, dtype=self.dtype)
            
            with torch.no_grad():
                u0 = self.ednn(x0, params_pre, hidden = False)   # 事前モデルでの出力
            
            u0 = u0.to(device=self.device, dtype=self.dtype)


            # パラメータの初期化
            params = self.ednn.get_init_params().to(device=self.device, dtype=self.dtype)
            params = nn.Parameter(params)

            # 最適化手法の設定
            itr = 0
            optimizer = torch.optim.Adam([params], lr=min(lr, 1e-3), weight_decay=0)
            

            # プログレスバーを設定
            with tqdm(total=max_itr, desc="Re-training Progress", unit="step") as pbar:
                # 最適化
                with torch.set_grad_enabled(True):
                    def closure():
                        nonlocal itr
                        optimizer.zero_grad()   # 勾配の初期化
                        loss = nn.functional.mse_loss(self.ednn(x0, params, hidden=False), u0)    # 損失の計算
                        if reg > 0.0:
                            loss = loss + reg * params.__pow__(2).sum()  # 正則化項の追加
                        loss.backward()  # 逆伝播
                        itr += 1

                        # プログレスバーの更新
                        pbar.set_postfix(loss=f"{loss.item():.6e}")  # 損失値を表示
                        pbar.update(1)  # プログレスバーを1ステップ進める
                        return loss.item()

                    # 最適化手法の選択
                    if optim.startswith("adam"):
                        self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training: IC by Adam...")
                        while itr < max_itr:
                            loss = closure()
                            optimizer.step()
                            if loss < atol: # 損失が許容誤差以下になったら終了
                                break
                    elif optim.startswith("lbfgs"):
                        self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training: IC by Adam first...")
                        while itr < max_itr // 10:
                            loss = closure()
                            optimizer.step()
                        self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training: IC by LBFGS...")
                        optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=max_itr - max_itr // 10, tolerance_grad=atol, tolerance_change=0, history_size=100)
                        optimizer.step(closure)
                    else:
                        raise NotImplementedError(self.optim)

            loss = nn.functional.mse_loss(self.ednn(x0, params), u0)
            self.logger(f"[{datetime.datetime.now()}] EDNN, Re-training: IC finished, Loss: {loss.item():.6e}")
            
            return params.data



    #   ODEの右辺を計算
    def forward(self, t, state):
        if self.nfe % self.log_freq == 0 or True:
            assert t.numel() == 1
            self.logger(f"[{datetime.datetime.now()}] EDNN, Integration: time={t.item():.6e}, nfe={self.nfe}, state mean={state.mean()}, var={state.var()}")
        self.nfe += 1
        if self.nfe % self.restart_fleq == 0 and self.nfe != 40000:
            #print("state_shape", state.shape)
            params_pre = state.reshape(self.params_shape)
            params_new = self.retraining(params_pre, reg=0.0, optim="adam", lr=1e-3, atol=1e-7, max_itr=20000)
            state = params_new.reshape(1, -1)
        
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

    