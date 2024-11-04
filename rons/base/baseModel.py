import os
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from .networks import get_network

# ベースモデルを抽象クラスとして定義
class BaseModel(ABC):
    # コンストラクタ
    def __init__(self, cfg):
        self.cfg = cfg  # 設定ファイル
        self.exp_dir = cfg.exp_dir  # 実験ディレクトリ 
        self.dt = cfg.dt    # 時間ステップ幅
        self.max_n_iters = cfg.max_n_iters  # 最大イテレーション数
        self.sample_resolution = cfg.sample_resolution  # サンプリング解像度
        self.vis_resolution = cfg.vis_resolution    # 可視化解像度
        self.timestep = -1  # 現在の時間ステップ
        
        self.tb = None  # TensorBoard用のオブジェクト
        self.min_lr = 1.1e-8    # 学習率の最小値
        self.early_stop_plateau = 500   # early stoppingのイテレーション数
        self.train_step = 0 # 現在の学習ステップ

        self.device = torch.device("cuda:0")    # デバイスの設定


    # ネットワークの初期化
    def _create_network(self, input_dim, output_dim):
        return get_network(self.cfg, input_dim, output_dim).to(self.device)
    
    # ネットワークを辞書形式で返す
    @property
    @abstractmethod
    def _trainable_networks(self):
        raise NotImplementedError
    
    # 学習中にデータポイントをサンプリングする抽象メソッド
    @abstractmethod
    def _sample_in_training(self):
        raise NotImplementedError
    
    # 初期条件(t=0)を学習させる抽象メソッド
    @abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    # タイムステップを進める抽象メソッド
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    # 出力をファイルに書き込む抽象メソッド
    def write_output(self, output_folder):
        pass
    
    # optimizer, schedulerを作成する
    def _reset_optimizer(self, use_scheduler=True, gamma=0.1, patience=500, min_lr=1e-8):
        param_list = [] # optimizerに渡すパラメータリスト
        
        # _trainable_networksからパラメータと学習率をリストに追加
        for net in self._trainable_networks.values():   
            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        
        # Adam optimizerをparam_listで初期化
        self.optimizer = torch.optim.Adam(param_list)
        
        # schedulerを用いる場合、ReduceLROnPlateauを初期化
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, min_lr=min_lr, patience=patience, verbose=True) if use_scheduler else None
    
    
    # TensorBoardのログを作成
    def _create_tb(self, name, overwrite=True):
        
        # self.cfg.log_dirにnameを追加してlog_pathを作成
        self.log_path = os.path.join(self.cfg.log_dir, name)
        
        # log_pathが存在する場合、上書き
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)    # 既存のlog_passを削除
        
        # TensorBoardのSummaryWriterを作成し、self.tbに格納
        if self.tb is not None:
            self.tb.close()
        self.tb = SummaryWriter(self.log_path)

    # ネットワークの重みをback propagationを用いて更新
    def _update_network(self, loss_dict):
        loss = sum(loss_dict.values())  # 損失の合計を計算
        self.optimizer.zero_grad()  # optimizerの勾配を初期化
        loss.backward() # back propagation
        
        # optimizerを用いてネットワークの重みを更新
        self.optimizer.step()
        
        # schedulerが存在する場合、schedulerを更新
        if self.scheduler is not None:
            self.scheduler.step(loss_dict['main'])

    # ネットワークの勾配を計算するかどうかを設定
    def _set_require_grads(self, model, require_grad: bool):
        for p in model.parameters():
            p.requires_grad_(require_grad)
    
    @classmethod
    def _timestepping(cls, func):
        def warp(self):
            self.timestep += 1
            self._create_tb(f"t{self.timestep:03d}")
            func(self)
            self.save_ckpt()
        return warp

    @classmethod
    def _training_loop(cls, func):
        """a decorator function that warps a function inside a training loop

        Args:
            func (_type_): a function that returns a dict of losses, must have key "main".
        """
        tag = func.__name__
        def loop(self, *args, **kwargs):
            pbar = tqdm(range(self.max_n_iters), desc=f"{tag}[{self.timestep}]")
            self._reset_optimizer()
            min_loss = float("inf")
            accum_steps = 0
            self.train_step = 0
            for i in pbar:
                # one gradient descent step
                loss_dict = func(self, *args, **kwargs)
                self._update_network(loss_dict)
                self.train_step += 1

                loss_value = {k: v.item() for k, v in loss_dict.items()}

                self.tb.add_scalars(tag, loss_value, global_step=i)
                pbar.set_postfix(loss_value)

                # optional visualization on tensorboard
                if (i == 0 or (i + 1) % self.cfg.vis_frequency == 0) and hasattr(self, f"_vis{tag}"):
                    vis_func = getattr(self, f"_vis{tag}")
                    vis_func()

                # early stop when converged
                if loss_value["main"] < min_loss:
                    min_loss, accum_steps = loss_value["main"], 0
                else:
                    accum_steps += 1

                if self.cfg.early_stop and self.optimizer.param_groups[0]['lr'] <= self.min_lr:
                    tqdm.write(f"early stopping at iteration {i}")
                    break
        return loop

    def save_ckpt(self, name=None):
        """save checkpoint for future restore"""
        if name is None:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{self.timestep:03d}.pth")
        else:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")

        save_dict = {}
        for name, net in self._trainable_networks.items():
            save_dict.update({f'net_{name}': net.cpu().state_dict()})
            net.cuda()
        save_dict.update({'timestep': self.timestep})

        torch.save(save_dict, save_path)
    
    def load_ckpt(self, name):
        """load saved checkpoint"""
        if type(name) is int:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{name:03d}.pth")
        else:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path)

        for name, net in self._trainable_networks.items():
            net.load_state_dict(checkpoint[f'net_{name}'])
        self.timestep = checkpoint['timestep']


















