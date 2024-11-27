import scipy
import numpy as np
import torch

from .dataset import BASEDataset

class Dataset(BASEDataset):

    def __init__(self, base_dir: str, N0: int = 0):
        # 定数の定義
        c = 1.0  # 移流速度
        Nx = 10000  # 空間方向のポイント数
        Nt = 201  # 時間方向のポイント数

        # 空間・時間の離散化
        x = np.linspace(-1.0, 1.0, Nx)
        t = np.linspace(0.0, 0.5, Nt)

        # メッシュグリッドの作成
        X, T = np.meshgrid(x, t)

        # 初期条件の定義
        def u0(x):
            return np.sin(np.pi * x)

        # 真の解の計算（解析解）
        U = u0(X - c * T)

        # データの辞書を作成
        data = {
            "x": x,
            "t": t,
            "uu": U
        }

        # クラスの属性を設定
        self.space_dim = 1
        self.state_dim = 1
        self.x_range = np.array([[-1.0, 1.0]])  # 空間範囲
        self.t_range = np.array([0.0, 1.0])     # 時間範囲

        self.data_x = data["x"].reshape(-1, 1)  # 形状 (512, 1)
        self.t_freq = Nt - 1  # 評価用の時間頻度
        self.data_u = data["uu"][:, :, None]   

        # 初期条件のサンプリング
        if N0 < 1:
            self.x0 = self.data_x                # 形状 (512, 1)
            self.u0 = self.data_u[0]             # 形状 (512, 1)
        else:
            idx_x = np.random.choice(self.data_x.shape[0], N0, replace=False)
            self.x0 = self.data_x[idx_x]
            self.u0 = self.data_u[0, idx_x]

        # 境界条件の設定
        self.is_zero_boundary = False
        self.is_periodic_boundary = True

    def get_initial_condition(self):
        return self.x0, self.u0

    def get_evaluation_data(self):
        return self.t_range, self.t_freq, self.data_x, self.data_u

    def equation(self, u, x):
        c = 1.0  # 移流速度

        # 自動微分を用いて空間微分を計算
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # 移流方程式の右辺を計算
        u_t = -c * u_x
        return u_t
    
    def initial_condition(self, x):
        return self.u0(x)
    
