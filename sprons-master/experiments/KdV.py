import scipy
import numpy as np
import torch

from .dataset import BASEDataset


class Dataset(BASEDataset):

    def __init__(self, base_dir: str, N0: int = 0):
        data = scipy.io.loadmat(f"{base_dir}/datasets/KdV.mat")
        self.space_dim = 1
        self.state_dim = 1
        self.x_range = np.array([[-1.0, 1.0]])  # x: 1x512 points, np.linspace(-1,1,513)[:-1]
        self.t_range = np.array([0.0, 1.0])  # t: 1x201 points, around np.linspace(0,1,201)

        # n_col x dim(x)
        self.data_x = np.array(data["x"].flatten())[:, None]
        # self.data_t = np.array(data["tt"].flatten())
        self.t_freq = 200
        self.data_u = np.array(data["uu"].T[:, :, None])  # (t,x,dim(u))-space

        if N0 < 1:
            self.x0 = self.data_x
            self.u0 = self.data_u[0]
        else:
            idx_x = np.random.choice(self.data_x.shape[0], N0, replace=False)  # N0 points from x
            self.x0 = self.data_x[idx_x]
            self.u0 = self.data_u[0, idx_x]

        self.x0 = np.linspace(self.x_range[0, 0], self.x_range[0, 1], 10001)[:-1].reshape(-1, 1)
        self.u0 = np.cos(np.pi * self.x0)

        self.is_zero_boundary = False
        self.is_periodic_boundary = True
        self.initial_condition = None

    def get_initial_condition(self):
        return self.x0, self.u0

    def get_evaluation_data(self):
        return self.t_range, self.t_freq, self.data_x, self.data_u

    def equation(self, u, x):
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
        u_t = -1.0 * u * u_x - 0.0025 * u_xxx

        # nabla_H = -0.5 * u**2 - 0.0025 * u_xx
        # u_t = torch.autograd.grad(nabla_H.sum(), x, create_graph=True)[0]

        return u_t
