import numpy as np
import torch

class BASEDataset:
    def __init__(self, base_dir, N0=None):
        self.is_zero_boundary = False
        self.is_periodic_boundary = False

    def get_initial_condition(self):
        x0 = np.array([])
        u0 = np.array([])
        return x0, u0

    def get_evaluation_data(self):
        data_x = np.array([])
        data_t = np.array([])
        data_u = np.array([])
        return data_x, data_t, data_u

    def u_t(self, u, x):
        return torch.tensor([])
