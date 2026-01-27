
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from .config import Config
# from utils.min_regressor_pytorch_matrix_pro import min_regressor_pytorch_matrix as W_min_matrix
from pinn_excitation_traj import W_min_matrix, Config

import numpy as np
import math

# ----------------------------
# Condition number objective (log cond)
# ----------------------------
def log_condition_number(Phi, eps=1e-8):
    # Use SVD to get singular values
    # torch.linalg.svdvals returns descending singular values
    s = torch.linalg.svdvals(Phi)
    s_max = s[0]
    s_min = s[-1]
    cond = (s_max + eps) / (s_min + eps)
    return torch.log(cond + 1e-12), cond

def stable_condition_number(matrix, eps=1e-7):
    """
    计算矩阵的条件数，避免除以过小的奇异值导致数值不稳定。
    matrix: Tensor，形状 (M, N)
    eps: float，最小奇异值的下限平滑项
    
    返回：条件数标量Tensor，支持反向传播
    """
    S = torch.linalg.svdvals(matrix)
    sigma_max = S[0]
    sigma_min = S[-1]

    # 防止除以零或非常小的数
    sigma_min_safe = sigma_min.clamp(min=eps)
    
    cond = sigma_max / sigma_min_safe
    return cond

# 自定义 sin 激活函数
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# def get_activation(act):
#     if act == "tanh":
#         return nn.Tanh()
#     elif act == "relu":
#         return nn.ReLU()
#     elif act == "sin":
#         return Sin()
#     else:
#         raise ValueError(f"Unknown activation {act}")

def get_activation(act):
    act = act.lower()

    if act == "relu":
        return nn.ReLU()

    elif act == "sigmoid":
        return nn.Sigmoid()

    elif act == "tanh":
        return nn.Tanh()

    elif act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    elif act == "elu":
        return nn.ELU()

    elif act in ["silu", "swish"]:
        return nn.SiLU()          # Swish = SiLU

    elif act == "gelu":
        return nn.GELU()

    elif act == "softplus":
        return nn.Softplus()

    elif act == "mish":
        return nn.Mish()

    elif act == "sin":
        return Sin()              # 你自定义的周期激活

    else:
        raise ValueError(
            f"Unknown activation '{act}'. "
            "Available: relu, sigmoid, tanh, leaky_relu, elu, silu/swish, gelu, softplus, mish, sin"
        )

"""
1.ReLu 2.Sigmoid 3. Tanh 4. LeakyReLU 5. ELU 6. Swish 7. GELU  8. softplus 9.Mish

分为四大类：

-分段线性 ReLU, Leaky ReLU      MLP CNN
-饱和型   Sigmoid, Tanh         RNN PINN
-平滑 ReLU ELU, Softplus        PINN  
-自门控平滑 Swish, GELU, Mish    Transformer  现代 CNN


✅ 存在三阶及以上导数（C³ 甚至 C∞）的激活函数
Sigmoid / Tanh / Softplus / Swish(SiLU) / GELU / Mish

❌ 不具备三阶导数（在某点不可导或高阶导数不存在）
ReLU / Leaky ReLU / ELU

"""


###############################################################
# PINN 模型
###############################################################
class PINN_model(nn.Module):
    def __init__(self, dof, q_min, q_max, hidden=(128, 128), activation="tanh"):
        super().__init__()
        self.dof = dof
        self.q_min = q_min
        self.q_max = q_max

        act = get_activation(activation)
        print(f"Using activation function: {act}")

        layers = []
        input_dim = 1
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(act)
            input_dim = h
        layers.append(nn.Linear(input_dim, dof))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        x = self.net(t)
        q = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(x)
        return q
    
class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, use_residual=False, norm_type=None, use_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        if use_norm and norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif use_norm and norm_type == "layer":
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = None

        self.activation = activation
        self.use_residual = use_residual and (in_features == out_features)

    def forward(self, x):
        out = self.linear(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.use_residual:
            out = out + x
        return out
class PINN_model_pro(nn.Module):
    def __init__(self, dof, q_min, q_max, hidden=(128, 128), activation="tanh",
                 use_residual=False, norm_type=None):
        super().__init__()
        self.dof = dof
        self.q_min = q_min
        self.q_max = q_max

        act_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            "sin": torch.sin
        }
        act = get_activation(activation)
        print(f"Using activation function: {act}")

        layers = []
        input_dim = 1
        for i, h in enumerate(hidden):
            use_norm = (i != 0)  # 第一个 block 不做归一化
            layers.append(DenseBlock(input_dim, h, act,
                                     use_residual=use_residual,
                                     norm_type=norm_type,
                                     use_norm=use_norm))
            input_dim = h
        layers.append(nn.Linear(input_dim, dof))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        x = self.net(t)
        q = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(x)
        return q

    def compute_derivatives(self, t):
        """
        输入:
            t: 时间张量，shape (M, 1)
        输出:
            q_pred: 关节角度预测 (M, dof)
            dq_pred: 关节速度预测 (M, dof)
            ddq_pred: 关节加速度预测 (M, dof)
        """
        q_pred = self.forward(t)
        dt = t[1] - t[0]

        dq_diff = torch.diff(q_pred, dim=0) / dt
        dq_pred = torch.cat([dq_diff, dq_diff[-1:].clone()], dim=0)

        ddq_diff = torch.diff(dq_diff, dim=0) / dt
        ddq_pred = torch.cat([ddq_diff, ddq_diff[-1:].clone(), ddq_diff[-1:].clone()], dim=0)

        return q_pred, dq_pred, ddq_pred


###############################################################
# 创建 Scheduler（根据 config）
###############################################################
def build_scheduler(optimizer, cfg):
    if not cfg.use_scheduler:
        return None

    if cfg.scheduler_type == "Cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
        )

    return None


###############################################################
# 训练函数
###############################################################
def train(cfg: Config,position=0):
    device = cfg.device
    dq_max = cfg.dq_max.to(device)
    ddq_max = cfg.ddq_max.to(device)
    dof=cfg.dof
    if cfg.model_type == "basic":
        model = PINN_model(
            dof=cfg.dof,
            q_min=cfg.q_min.to(device),
            q_max=cfg.q_max.to(device),
            hidden=cfg.hidden,
            activation=cfg.activation,
        ).to(device)
    elif cfg.model_type == "pro":
        model = PINN_model_pro(
            dof=cfg.dof,
            q_min=cfg.q_min.to(device),
            q_max=cfg.q_max.to(device),
            hidden=cfg.hidden,
            activation=cfg.activation,
            use_residual=cfg.use_residual,
            norm_type=cfg.norm_type,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    t_samples = torch.linspace(0, cfg.T, cfg.M, device=device).unsqueeze(1)

    # Optimizer
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "LBFGS":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=cfg.lr,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )
    else:
        raise ValueError("Unknown optimizer")

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)

    # pbar = tqdm(range(1, cfg.epochs + 1))
    # pbar = tqdm(range(1, cfg.epochs + 1), position=position, leave=True)
    pbar = tqdm(
        range(1, cfg.epochs+1),
        position=position, leave=True,
        desc=f"Traj {position}"
    )
   # ===== before training loop =====
    if cfg.save_loss:
        Loss = {
            "loss": [],
            "cond": [],
            "boundary": [],
            "bounds": [],
            "smooth": [],
        } 

    for it in pbar:

        t = t_samples.clone().detach().requires_grad_(True)
        loss_dict = {}
        def closure():
            optimizer.zero_grad()
            q_pred = model(t)
            
            # 计算损失，下面示例是你已有的loss计算逻辑，简化版
            # 你需要把loss的计算和backward写在这里
            # 这里写你的loss计算
            dt = t[1] - t[0]
            dq_diff = torch.diff(q_pred, dim=0) / dt
            dq_pred = torch.cat([dq_diff, dq_diff[-1:].clone()], dim=0)
            ddq_diff = torch.diff(dq_diff, dim=0) / dt
            ddq_pred = torch.cat([ddq_diff, ddq_diff[-1:].clone(), ddq_diff[-1:].clone()], dim=0)

            Phi = W_min_matrix(q_pred, dq_pred, ddq_pred)
            logcond, cond_val = log_condition_number(Phi, eps=cfg.eps)

            q0, qf = q_pred[0], q_pred[-1]
            dq0, dqf = dq_pred[0], dq_pred[-1]
            ddq0, ddqf = ddq_pred[0], ddq_pred[-1]

            boundary_loss = (
                q0.norm() + qf.norm() +
                dq0.norm() + dqf.norm() +
                ddq0.norm() + ddqf.norm()
            )

            dq_violation = torch.relu(torch.abs(dq_pred) - dq_max)
            ddq_violation = torch.relu(torch.abs(ddq_pred) - ddq_max)
            bounds_loss = dq_violation.sum() + ddq_violation.sum()

            d3q = torch.diff(ddq_pred, dim=0) / dt
            smooth_loss = (d3q ** 2).sum() / (cfg.M - 1)

            loss = (
                cond_val +
                cfg.w_boundary * boundary_loss +
                cfg.w_bounds * bounds_loss +
                cfg.w_smooth * smooth_loss
            )

            loss.backward()
            loss_dict['loss'] = loss.detach()
            loss_dict['cond_val'] = cond_val.detach()
            loss_dict['boundary_loss'] = boundary_loss.detach()
            loss_dict['bounds_loss'] = bounds_loss.detach()
            loss_dict['smooth_loss'] = smooth_loss.detach()
            return loss

        if cfg.optimizer == "LBFGS":
            optimizer.step(closure)
            loss = loss_dict['loss']
            lr_current =cfg.lr  # LBFGS 不更新 lr
            cond_val = loss_dict['cond_val']
            boundary_loss = loss_dict['boundary_loss']
            bounds_loss = loss_dict['bounds_loss']
            smooth_loss = loss_dict['smooth_loss']

        else:
            q_pred = model(t)

            # d_q = []
            # dd_q = []
            # dt = t[1] - t[0]
            # for j in range(dof):
            #     qj = q_pred[:, j].unsqueeze(1)  # [M,1]
            #     dq = torch.autograd.grad(qj, t, torch.ones_like(qj), create_graph=True, retain_graph=True)[0]  # [M,1]
            #     ddq = torch.autograd.grad(dq, t, torch.ones_like(dq), create_graph=True, retain_graph=True)[0]
            #     d_q.append(dq.squeeze(1))
            #     dd_q.append(ddq.squeeze(1))
            # dq_pred = torch.stack(d_q, dim=1)   # [M,dof]
            # ddq_pred = torch.stack(dd_q, dim=1) # [M,dof]

            dt = t[1] - t[0]
            dq_diff = torch.diff(q_pred, dim=0) / dt
            dq_pred = torch.cat([dq_diff, dq_diff[-1:].clone()], dim=0)
            ddq_diff = torch.diff(dq_diff, dim=0) / dt
            ddq_pred = torch.cat([ddq_diff, ddq_diff[-1:].clone(), ddq_diff[-1:].clone()], dim=0)

            # W_min_matrix_jit = torch.jit.trace(W_min_matrix, (q_pred, dq_pred, ddq_pred))
            # Phi = W_min_matrix_jit(q_pred, dq_pred, ddq_pred)
            Phi = W_min_matrix(q_pred, dq_pred, ddq_pred)
            logcond, cond_val = log_condition_number(Phi, eps=cfg.eps)


            q0, qf = q_pred[0], q_pred[-1]
            dq0, dqf = dq_pred[0], dq_pred[-1]
            ddq0, ddqf = ddq_pred[0], ddq_pred[-1]

            boundary_loss = (
                q0.norm() + qf.norm() +
                dq0.norm() + dqf.norm() +
                ddq0.norm() + ddqf.norm()
            )

            dq_violation = torch.relu(torch.abs(dq_pred) - dq_max)
            ddq_violation = torch.relu(torch.abs(ddq_pred) - ddq_max)
            bounds_loss = dq_violation.sum() + ddq_violation.sum()

            d3q = torch.diff(ddq_pred, dim=0) / dt
            smooth_loss = (d3q ** 2).sum() / (cfg.M - 1)

            loss = (
                cond_val +
                cfg.w_boundary * boundary_loss +
                cfg.w_bounds * bounds_loss +
                cfg.w_smooth * smooth_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()


            lr_current = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': loss.item() if torch.is_tensor(loss) else loss,
            'cond': cond_val.item(),
            'lr': f'{lr_current:.2e}',
            'boundary': boundary_loss.item(),
            'bounds': bounds_loss.item(),
            'smooth': smooth_loss.item()
        })
        # ===== record loss (controlled) =====
        if cfg.save_loss:
            Loss["loss"].append(loss.item())
            Loss["cond"].append(cond_val.item())
            Loss["boundary"].append(boundary_loss.item())
            Loss["bounds"].append(bounds_loss.item())
            Loss["smooth"].append(smooth_loss.item())

        if cfg.save_model:
            window_save_best(model, cond_val, boundary_loss, bounds_loss, it, position=position)


    # ===== return =====
    if cfg.save_loss:
        return model, Loss
    else:
        return model
    



import time
import os
import copy
def window_save_best(model, cond, boundary_loss, bounds_loss, it,
                     win=2000, boundary_eta=5e-2, bounds_eta=1e-3, position=0):

    # static state
    if not hasattr(window_save_best, "best"):
        window_save_best.best = float("inf")
        window_save_best.state = None
        window_save_best.idx = 1
        window_save_best.it = 1
        window_save_best.start_time = time.time()
        window_save_best.best_time = None


    k = (it - 1) % win + 1

    # update best inside window
    if boundary_loss.item() < boundary_eta and bounds_loss.item() < bounds_eta:
        if cond < window_save_best.best:
            window_save_best.best = float(cond)
            # window_save_best.state = model.state_dict()
            window_save_best.state = copy.deepcopy(model.state_dict())
            window_save_best.it = it
            window_save_best.best_time = time.time()
            
    # end of window → save & reset
    if k == win:
        if window_save_best.state is not None:
            # 确保models目录存在
            save_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(save_dir, exist_ok=True)

            path = os.path.join(save_dir, f"pinn_traj_{position}_{window_save_best.idx}.pt")
            # torch.save(window_save_best.state, path)
            torch.save(window_save_best.state, path, _use_new_zipfile_serialization=True)
            elapsed = window_save_best.best_time - window_save_best.start_time

            # ===== write metrics =====
            txt_path = path.replace(".pt", ".txt")
            with open(txt_path, "w") as f:
                # f.write(f"best_cond      : {window_save_best.best:.6e}\r")
                # f.write(f"boundary_loss  : {boundary_loss.item():.6e}\r")
                # f.write(f"bounds_loss    : {bounds_loss.item():.6e}\r")
                # f.write(f"iteration      : {it}\r")

                f.write(
                    f"traj={position} "
                    f"iter={window_save_best.it} "
                    f"best_cond={window_save_best.best:.6e} "
                    f"boundary={boundary_loss.item():.6e} "
                    f"bounds={bounds_loss.item():.6e} "
                    f"elapsed_sec={elapsed:.2f}\n"
                )

            print(
                f"[SAVE] pinn_traj{position}_{window_save_best.idx}.pt "
                f"best_cond={window_save_best.best:.2e}"
            )

            window_save_best.idx += 1

        window_save_best.best = float("inf")
        window_save_best.state = None
        window_save_best.state = None
        window_save_best.best_time = None


from multiprocessing import Process

def worker(traj_id):
    cfg = Config()
    # optional: 设置不同随机种子
    # cfg.seed = traj_id
    train(cfg, position=traj_id)

def multi_train(n_jobs=8):
    processes = []
    for traj_id in range(n_jobs):
        p = Process(target=worker, args=(traj_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()