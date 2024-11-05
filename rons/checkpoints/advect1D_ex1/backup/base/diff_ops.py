# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py
import torch
from torch.autograd import grad

# yのxに関するヘッセ行列
def hessian(y, x):
    """
    Hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    meta_batch_size, num_observations = y.shape[:2] # バッチサイズと観測数を取得
    grad_y = torch.ones_like(y[..., 0]).to(y.device)    # 勾配の初期化
    h = torch.zeros(meta_batch_size, num_observations,
                    y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device) # ヘッセ行列を格納するためのゼロテンソルh
    
    for i in range(y.shape[-1]):
        # yの各特徴のxに関する勾配dydxを計算、計算グラフを保存
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # dydxのj番目の成分のxに関する勾配を計算、計算グラフを保存
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y,
                                   create_graph=True)[0][..., :]

    status = 0  # 計算結果にNaNが含まれているかのフラグ
    if torch.any(torch.isnan(h)):
        status = -1 # NaNが含まれていればstatus=-1
    return h, status


# yのxに関するラプラシアン∆y
def laplace(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)   # yのxに関する勾配を計算
    
    # normalizeがTrueの場合、正規化を行う
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)   # epsは0除算を防ぐため
    
    # gradの発散（ラプラシアン）を計算
    div = divergence(grad, x)

    #   return_gradがTrueの場合、gradも返す
    if return_grad:
        return div, grad
    return div

# yのxに関する発散∇・y
def divergence(y, x):
    div = 0.
    
    # yの各成分に関してループ
    for i in range(y.shape[-1]):
        # yの第i成分をxで微分した結果をdivに加算
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div

# yのxに関する勾配∇y
def gradient(y, x, grad_outputs=None):
    # grad_outputsがNoneの場合、yと同じ形状かつ要素が全て１
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    
    # 勾配計算
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# yのxに関するヤコビ行列
def jacobian(y: torch.FloatTensor, x: torch.FloatTensor):

    """jacobian of y wrt x

    Args:
        y (torch.FloatTensor): (N, dim_y)
        x (torch.FloatTensor): (N, dim_x)

    Returns:
        jac (torch.FloatTensor): (N, dim_y, dim_x)
    """
    # 出力テンソルの形状のゼロテンソルjacを作成
    jac = torch.zeros(*y.shape[:-1], y.shape[-1], x.shape[-1]).to(y.device)

    # yの各出力次元iについてヤコビ計算
    for i in range(y.shape[-1]):
        y_i = y[..., i]
        jac[..., i, :] = grad(y_i, x, torch.ones_like(y_i), create_graph=True)[0]

    status = 0  # 計算結果にNaNが含まれているかのフラグ
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
