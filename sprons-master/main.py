import torch
import torch.backends.cudnn
import torch.backends.cuda

torch.backends.cudnn.deterministic = False  # Trueで再現性が高まるが遅くなる
torch.backends.cudnn.benchmark = True   # Trueでアルゴリズム選択の最適化を行う
torch.backends.cuda.matmul.allow_tf32 = False   # TrueでTF32(低精度高速)を許可
torch.backends.cudnn.allow_tf32 = False # TrueでCuDNNにTF32(低精度高速)を許可
torch.set_grad_enabled(False)   # 勾配計算を行わない
torch.set_default_dtype(torch.float64)  # デフォルトのテンソル型を設定
import numpy as np
import os
import sys
import argparse
import experiments
import pickle
import datetime

import experiments.model

# コマンドライン引数の設定
def get_args():
    parser = argparse.ArgumentParser(description=None)
    # network
    parser.add_argument("--model", default="fc", type=str, help="NN model.")
    parser.add_argument("--dim_hidden", default=30, type=int, help="number of hidden units.")
    parser.add_argument("--num_layers", default=4, type=int, help="number of hidden layers.")
    parser.add_argument("--act", default="silu", type=str, help="activation function.", choices=["tanh", "sin", "silu", "relu"])
    parser.add_argument("--sinusoidal", default=5, type=int, help="sinusoidal embedding.")
    parser.add_argument("--head_tuning", default=1, type=int, help="head tuning.", choices=[0, 1])
    # experiments
    parser.add_argument("--ireg", default=0.0, type=float, help="regularizer for parameters in initialization.")
    parser.add_argument("--preg", default=0.0, type=float, help="regularizer for parameters in prediction.")
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument("--dataset", default="AC", type=str, help="dataset name")
    parser.add_argument("--optim", default="lbfgs", type=str, help="optimizer for initialization.", choices=["adam", "lbfgs"])
    parser.add_argument("--max_itr", default=50000, type=int, help="number of iterations for initialization.")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for initialization.")
    parser.add_argument("--itol", default=1e-9, type=float, help="absolute/gradient tolerance for initialization.")
    parser.add_argument("--hreg", default=1e-5, type=float, help="regularizer for parameters in head tuning.")
    # integration
    parser.add_argument("--method", default="PreCG", type=str, help="method to obtain gradient.", choices=["optimization", "inversion", "collocation", "gelsd", "gpytorchCG", "PreCG"])
    parser.add_argument("--solver", default="rk4", type=str, help="numerical integrator.")
    parser.add_argument("--n_eval", default=2000, type=int, help="number of collocation points for solver. Set 0 to use the default points.")
    parser.add_argument("--atol", default=1e-3, type=float, help="absolute tolerance for solver.")
    parser.add_argument("--rtol", default=1e-4, type=float, help="relative tolerance for solver.")
    parser.add_argument("--substeps", default=50, type=int, help="number of substeps for solver.")
    parser.add_argument("--restart_fleq",default=4000, type=int, help="restart frequency for EDNN.")
    # display
    parser.add_argument("--log_freq", default=200, type=int, help="number of steps between prints.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="verbose?.")
    parser.add_argument("--experiment_dir", default="experiments", type=str, help="where to save the trained model.")
    parser.add_argument("--result_dir", default="results", type=str, help="where to save the results.")
    parser.add_argument("--postfix", default=None, type=str, help="postfix for saved files.")
    parser.add_argument("--noretry", dest="noretry", action="store_true", help="not perform a finished trial.")
    parser.add_argument("--mark_running", dest="mark_running", action="store_true", help="block performing a running trial.")
    # parser.add_argument("--norm", dest="norm", action="store_true", help="data normalization at the first layer.")
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    # 実験設定の識別ラベル
    label = f"{args.dataset}-{args.method}-{args.solver}"
    label += f"-ireg{args.ireg}" if args.ireg > 0 else ""
    label += f"-preg{args.preg}" if args.preg > 0 else ""
    #label += f"-atol{args.atol}"
    #label += f"-rtol{args.rtol}"
    label += f"-sinusoidal{args.sinusoidal}"
    label += f"-headtuning{args.head_tuning}"
    label += f"-{args.optim}" 
    label += f"-{args.restart_fleq}"
    label += f"-{args.postfix}" if args.postfix else ""
    label += f"-seed{args.seed}"
    args.result_path = f"{args.experiment_dir}/{args.result_dir}/{label}"
    args.path_txt = f"{args.result_path}.txt"

    return args


def main(args):
    # load data
    import importlib

    dataset = importlib.import_module(f"{args.experiment_dir}.{args.dataset}").Dataset(args.experiment_dir)
    x0, u0 = dataset.get_initial_condition()    # initial condition
    t_range, t_freq, data_x, data_u = dataset.get_evaluation_data() # evaluation data
    
    # model
    ednn = experiments.model.EDNN(
        x_range=dataset.x_range,    # xの範囲
        space_dim=dataset.space_dim,    # 空間次元
        state_dim=dataset.state_dim,    # 状態次元
        dim_hidden=args.dim_hidden,   # 隠れ層の次元
        num_layers=args.num_layers,   # 隠れ層の数
        nonlinearity=args.act,  # 活性化関数
        sinusoidal=args.sinusoidal, # サイン波埋め込み
        is_periodic_boundary=dataset.is_periodic_boundary,  # 周期境界条件
        is_zero_boundary=dataset.is_zero_boundary,  # ゼロ境界条件
        space_normalization=True,   # 空間の正規化
    )
    
    
    print(t_range, t_freq, args.substeps)

    # stream data
    if t_range[1] == int(t_range[1]) and t_freq == int(t_freq):
        data_t = np.linspace(0, t_range[-1], int(t_range[-1] * t_freq * args.substeps + 1))
    else:
        data_t = np.arange(0, t_range[-1] + t_range[-1] / (t_freq * args.substeps), t_range[-1] / (t_freq * args.substeps))

    # logger
    logging_file = open(f"{args.result_path}.log", "w")
    logger = lambda *args: print(*args) or print(*args, flush=True, file=logging_file)

    # training initial condition
    trainer = experiments.model.EDNNTrainer(ednn, log_freq=args.log_freq, restart_fleq=args.restart_fleq, logger=logger)
    params = trainer.learn_initial_condition(x0, u0, reg=args.ireg, optim=args.optim, lr=args.lr, atol=args.itol, max_itr=args.max_itr)
    
    # Head Tuning
    if args.head_tuning:
        params = trainer.head_tuning(params, x0_head, u0_head, args.hreg)
    
    # integration
    us, ps = trainer.integrate(params=params, equation=dataset.equation, method=args.method, solver=args.solver, t_eval=data_t, x_eval=data_x, n_eval=args.n_eval, reg=args.preg, atol=args.atol, rtol=args.rtol)

    # to numpy
    us, ps = us[:: args.substeps].detach().cpu().numpy(), ps[:: args.substeps].detach().cpu().numpy()
    
    # error
    error_course = (us - data_u).__pow__(2).mean(-1).mean(-1)

    return us, ps, error_course, data_u


if __name__ == "__main__":
    args = get_args()
    
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create results directory
    os.makedirs(f"{args.experiment_dir}/{args.result_dir}", exist_ok=True)
    
    # check if the trial is already done
    if os.path.exists(args.path_txt):
        if args.noretry:
            print(f"[{datetime.datetime.now()}] Manager: trial already done,", " ".join(sys.argv), flush=True)
            print(args.path_txt)
            exit()
            

    # mark the trial as running
    if args.mark_running:
        with open(args.path_txt, "w") as of:
            print("a running trial...", file=of)

    # print the trial
    print(f"[{datetime.datetime.now()}] Manager: trial not yet,", " ".join(sys.argv), flush=True)
    print(f"[{datetime.datetime.now()}] Manager: trial entering,", " ".join(sys.argv))

    # run the trial
    if args.mark_running:
        try:
            us, ps, error, data_u = main(args)
        except:
            with open(args.path_txt, "w") as of:
                print("a running trial...failure.", file=of)
            exit(-1)
    else:
        us, ps, error, data_u = main(args)

    from PIL import Image   # Python Imaging Library
    
    # normalize and save the results
    u_max = data_u.max()
    u_min = data_u.min()
    for i in range(us.shape[-1]):
        Image.fromarray(((us - u_min) / (u_max - u_min)).clip(0, 1).__mul__(256).astype(np.uint8)[..., i]).save(f"{args.result_path}_{i}_result.png")
    for i in range(data_u.shape[-1]):
        Image.fromarray(((data_u - u_min) / (u_max - u_min)).clip(0, 1).__mul__(256).astype(np.uint8)[..., i]).save(f"{args.result_path}_{i}_truth.png")

    with open(f"{args.result_path}.pkl", "wb") as handle:
        pickle.dump({"u": us, "params": ps, "error": error, "data_u": data_u}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(args.path_txt, error)

    print(f"[{datetime.datetime.now()}] Manager: trial ended,", " ".join(sys.argv), flush=True)
