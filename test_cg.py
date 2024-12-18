import torch

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
    
# weightsとbiasesを結合
def concat_params(self, weights, biases):
    params_list = []
    for w, b in zip(weights, biases):
        # weightをフラット化
        params_list.append(w.view(-1))
        # biasもフラット化
        params_list.append(b.view(-1))
        
    # 全てのパラメータをまとめて1つのテンソルに結合
    params = torch.cat(params_list)
    return params