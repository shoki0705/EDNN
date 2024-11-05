import torch
import torch.nn as nn
import numpy as np
from siren_pytorch import SirenNet

#   ニューラルネットワークの定義
def get_network(cfg, in_features, out_features):
    if cfg.network == 'MLP':
        return MLP(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features, nonlinearity=cfg.nonlinearity)
    elif cfg.network == 'siren':
        return SirenNet(dim_in = in_features, dim_hidden = cfg.hidden_features, dim_out = out_features,
            num_layers = cfg.num_hidden_layers, w0 = 30.0, w0_initial = 1)
    # Based on https://github.com/lucidrains/siren-pytorch
    else:
        raise NotImplementedError
    


# MLPモデル
class MLP(nn.Module):
    # コンストラクタ
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        # 活性化関数と初期化方法
        nls_and_inits = {
            'relu': (nn.ReLU(inplace=True), init_weights_normal),
            'elu': (nn.ELU(inplace=True), init_weights_elu)
        }
        
        # 活性化関数,重みの初期関数
        nl, nl_weight_init = nls_and_inits[nonlinearity]

        # 重みの初期化
        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        # ネットワークの構築
        self.net = []
        
        # 入力層を追加
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        # 隠れ層を追加
        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        # 出力層を追加
        self.net.append(nn.Linear(hidden_features, out_features))
        
        # 最後の層が線形層でない場合、指定された活性化関数を追加
        if not outermost_linear:
            self.net.append(nl)

        # ネットワークをシーケンシャルに変換
        self.net = nn.Sequential(*self.net)
        
        # 重みの初期化
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

    # 順伝播
    def forward(self, coords, weights=None):
        output = self.net(coords)
        if weights is not None:
            output = output * weights
        return output


#   ReLU用に重みを初期化
def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
            


#  ELU用に重みを初期化
def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))
















