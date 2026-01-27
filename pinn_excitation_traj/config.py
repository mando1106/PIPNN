# config.py
import math
import torch

class Config:
    # Device
    device = "cpu"
    model_type = "pro"
    use_residual = True            # 是否使用残差连接
    norm_type = "layer"            # None / "batch" / "layer"


    # Task
    dof = 6
    T = 20
    M = 1000

    # Joint limits
    pi = math.pi
    q_max = torch.tensor([pi, 0.5*pi, 2/3*pi, 2/3*pi, 2/3*pi, pi])
    q_min = -q_max

    dq_max = torch.tensor([0.8, 0.8, 0.8, 1.0, 1.0, 1.5])
    ddq_max = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Model
    activation = "sin"             # sin / tanh / relu
    hidden = (256, 256)

    # Training options
    optimizer = "Adam"             # "Adam" / "LBFGS"
    lr = 1e-3
    epochs = 2000

    # Scheduler settings
    use_scheduler = True           
    scheduler_type = "Cosine"      # "Cosine" / "None"
    eta_min = 1e-5                 # for CosineAnnealingLR

    # Loss weights
    w_boundary = 1e2
    w_bounds = 1e2
    w_smooth = 0.5e2

    # numerical stability
    eps = 1e-8

    # save models
    save_model = False
    save_loss  = True
