import os
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from .networks import get_network

# ベースモデルを抽象クラスとして定義
class BaseModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = cfg.exp_dir
        self.dt = cfg.dt
        self.max_n_iters = cfg.max_n_iters
        self.sample_resolution = cfg.sample_resolution
        self.vis_resolution = cfg.vis_resolution
        self.timestep = -1
        
        self.tb = None
        self.min_lr = 1.1e-8
        self.early_stop_plateau = 500
        self.train_step = 0

        self.device = torch.device("cuda:0")