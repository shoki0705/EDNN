import torch
from cola.ops import Sum, Dense

# サンプルデータ
d, k, p = 1, 2, 3  # dim(u), batch, dim(params)
J = torch.randn(d, k, p)
print(J)

# 元の einsum を用いた結果
M_einsum = torch.einsum("dki,dkj->dij", (J, J))

print(M_einsum.shape)

# Sum を用いた計算
components = [Dense(J[b_idx].T @ J[b_idx]) for b_idx in range(J.shape[0])]  # 各バッチで計算
M_sum = Sum(*components)

# 結果の確認
for b_idx in range(J.shape[0]):
    M_sum_result = components[b_idx]._matmat(torch.eye(p))  # Dense オブジェクトの計算
    print(f"Batch {b_idx}:")
    print("元の M_einsum:", M_einsum[b_idx])
    print("Sum を用いた M:", M_sum_result)
    print("差分:", torch.norm(M_einsum[b_idx] - M_sum_result))