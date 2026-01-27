
from .config import Config as BaseConfig   # ← 改名避免冲突
from .model_pinn import *

class Config(BaseConfig):
    # Device
    device = "cpu"                 # cpu cuda
    model_type = "pro"             # "pro"  "basic" 
    use_residual =  True           # 是否使用残差连接 False True
    norm_type = "layer"            # None / "batch" / "layer"

    # Task
    dof = 6
    T = 20
    M = 1000

    # Joint limits
    pi = math.pi
    q_max = torch.tensor([pi, 0.5*pi, 2/3*pi, 2/3*pi, 2/3*pi, pi])
    q_min = -q_max

    dq_max = torch.tensor([0.8, 0.8, 0.8, 0.8, 1.0, 1.5])
    ddq_max = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Model
    activation = "sin"             # sin / tanh / relu
    hidden = (256, 256)

    # Training options
    optimizer = "Adam"             # "Adam" / "LBFGS"
    lr = 1e-3                       # 1e-3 1e-4
    epochs = 2000

    # Scheduler settings
    use_scheduler = True           # True False
    scheduler_type = "Cosine"      # "Cosine" / "None"
    eta_min = 1e-5                 # for CosineAnnealingLR

    # Loss weights
    w_boundary = 1e2
    w_bounds = 1e2
    w_smooth = 0.5e2

    # numerical stability
    eps = 1e-8

# cfg = Config()
# model = train(cfg)

from multiprocessing import Process

# def worker(traj_id):
#     cfg = Config()
#     print(f"Running task {traj_id}")
#     model = train(cfg)

# if __name__ == "__main__":
#     processes = []
#     for i in range(8):  # 启动8个并行任务
#         p = Process(target=worker, args=(i,))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

def worker(traj_id):
    cfg = Config()
    # optional: 设置不同随机种子
    # cfg.seed = traj_id
    model,Loss =train(cfg, position=traj_id)

def multi_train(n_jobs=8):
    processes = []
    for traj_id in range(n_jobs):
        p = Process(target=worker, args=(traj_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":            
    import multiprocessing
    # multiprocessing.set_start_method("spawn")
    multi_train(8)