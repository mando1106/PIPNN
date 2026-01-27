function [g, h] = Nonlcon(x, dof, N, t_end, dt , f)
% -------------------------------------------------------
% 通用机械臂激励轨迹非线性约束函数
% 自动化、适配任意自由度与阶数
%
% 输入：
%   x     - 优化变量向量 [alpha, beta, delta]
%   dof   - 自由度
%   N     - alpha / beta 阶数
%   t_end - 轨迹周期时间 (s)
%   dt    - 采样时间步长 (可选，用于确定频率)
%
% 输出：
%   g     - 不等式约束 (g(x) <= 0)
%   h     - 等式约束 (h(x) == 0)
% -------------------------------------------------------



wf = 2 * pi * f;        % 基角频率

%% === 机械臂约束参数 ===
q_max   = [pi, 1/2*pi, 2/3*pi, 2/3*pi, 2/3*pi, pi];
q_min   = [-pi, -1/2*pi, -2/3*pi, -2/3*pi, -2/3*pi, -pi];
dq_max  = [0.8, 0.8, 0.8, 1.0, 1.0, 1.5];
ddq_max = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];



%% === 参数展开 ===
vars_per_joint = 2*N + 1;
alpha = zeros(dof, N);
beta  = zeros(dof, N);
delta = zeros(dof, 1);

for j = 1:dof
    idx = (j-1)*vars_per_joint;
    alpha(j,:) = x(idx + (1:N));
    beta(j,:)  = x(idx + (N+1 : 2*N));
    delta(j)   = x(idx + 2*N + 1);
end

k = (1:N)';  % 向量形式，方便矩阵计算

%% === 不等式约束 g(x) <= 0 ===
num_g = 4 * dof;  % 每个关节4个不等式约束
g = zeros(num_g, 1);
g_idx = 0;

for j = 1:dof
    amp = sqrt(alpha(j,:).^2 + beta(j,:).^2);
    
    % --- 角度上限 ---
    sum_term = sum(amp ./ (k'*wf));
    g_idx = g_idx + 1;
    g(g_idx) = sum_term + delta(j) - q_max(j);
    
    % --- 角度下限 ---
    g_idx = g_idx + 1;
    g(g_idx) = sum_term - delta(j) + q_min(j);
    
    % --- 角速度 ---
    sum_term = sum(amp);
    g_idx = g_idx + 1;
    g(g_idx) = sum_term - dq_max(j);
    
    % --- 角加速度 ---
    sum_term = wf * sum(k' .* amp);
    g_idx = g_idx + 1;
    g(g_idx) = sum_term - ddq_max(j);
end

%% === 等式约束 h(x) == 0 ===
% 每个关节6个等式约束: q(0)=δ, dq(0)=0, ddq(0)=0, q(tf)=0, dq(tf)=0, ddq(tf)=0
num_h = 6 * dof;
h = zeros(num_h, 1);
h_idx = 0;

sin_term = sin(k*wf*t_end);
cos_term = cos(k*wf*t_end);

for j = 1:dof
    % --- 初始约束 ---
    h_idx = h_idx + 1;  % q(0)
    h(h_idx) = sum(beta(j,:) ./ (k'*wf)) - delta(j);
    
    h_idx = h_idx + 1;  % dq(0)
    h(h_idx) = sum(alpha(j,:));
    
    h_idx = h_idx + 1;  % ddq(0)
    h(h_idx) = wf * sum(k' .* beta(j,:));
    
    % --- 终值约束 ---
    h_idx = h_idx + 1;  % q(tf)
    h(h_idx) = sum((alpha(j,:)./(k'*wf)) .* sin_term' - ...
                   (beta(j,:)./(k'*wf)) .* cos_term') - delta(j);
    
    h_idx = h_idx + 1;  % dq(tf)
    h(h_idx) = sum(alpha(j,:) .* cos_term' + beta(j,:) .* sin_term');
    
    h_idx = h_idx + 1;  % ddq(tf)
    h(h_idx) = wf * sum(k' .* (beta(j,:).*cos_term' - alpha(j,:).*sin_term'));
end

end
