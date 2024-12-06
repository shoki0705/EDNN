import numpy as np
import torch

from .dataset import BASEDataset


class Dataset(BASEDataset):

    def __init__(self, base_dir: str, N0: int = 0):
        c = 1.0
        Nx = 100
        Nt = 201

        x = np.linspace(-1.0, 1.0, Nx)
        y = np.linspace(-1.0, 1.0, Nx)
        t = np.linspace(0.0, 0.5, Nt)
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')

        u0 = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        
        
        self.initial_condition = u0
        
        q = [X,Y]

        U = u0(q) * np.cos(c * np.sqrt(2) * T)

        data = {"x": x, "y": y, "t": t, "uu": U}

        self.space_dim = 2
        self.state_dim = 1
        self.x_range = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        self.t_range = np.array([0.0, 0.5])

        self.data_x = np.stack([X[..., 0].flatten(), Y[..., 0].flatten()], axis=-1)
        self.t_freq = Nt - 1
        self.data_u = data["uu"][:, :, :, None]

        if N0 < 1:
            self.x0 = self.data_x
            self.u0 = self.data_u[:, :, 0].flatten()
        else:
            idx_x = np.random.choice(self.data_x.shape[0], N0, replace=False)
            self.x0 = self.data_x[idx_x]
            self.u0 = self.data_u[:, :, 0].flatten()[idx_x]

        self.u0 = self.u0[:, None]  # (N, 1) にリシェイプ

        self.is_zero_boundary = True
        self.is_periodic_boundary = False

    def get_initial_condition(self):
        return self.x0, self.u0

    def get_evaluation_data(self):
        return self.t_range, self.t_freq, self.data_x, self.data_u

    def equation(self, u, x, y):
        c = 1.0
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        laplacian_u = u_x + u_y
        u_tt = c**2 * laplacian_u
        return u_tt
