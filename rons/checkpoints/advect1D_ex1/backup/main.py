import os
from config import Config

# 実験設定
cfg = Config("train")

# モデルを選択
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

# resultsに結果を保存する
output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)

# 時間積分
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        model.initialize()
    else:
        model.step()
    
    # 現在の状態をoutput_folderに出力
    model.write_output(output_folder)
