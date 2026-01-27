
function [t, q, qd, qdd] = traj(x, dof, N, t_end, dt, f)
% generate_trajectory 基于Fourier级数生成机械臂轨迹（角度、速度、加速度）
%
% 输入：
%   x    - 参数向量，结构为每个关节：(N个alpha + N个beta + 1个offset)，共 dof*(2*N+1) 长度
%   dof  - 关节自由度数量
%   N    - Fourier阶数
%   t_end - 轨迹结束时间（秒）
%   dt   - 时间步长（秒）
%
% 输出：
%   t    - 时间向量 [1 x M]
%   q    - 关节角度矩阵 [dof x M]
%   qd   - 关节速度矩阵 [dof x M]
%   qdd  - 关节加速度矩阵 [dof x M]


t = 0:dt:t_end;    % 时间向量
wf = 2 * pi * f; % 基角频率，根据总周期定义

M = length(t);

% 校验参数长度是否匹配
expected_len = dof * (2*N + 1);
assert(length(x) == expected_len, ...
    sprintf('参数长度应为 %d，但收到 %d', expected_len, length(x)));

q = zeros(dof, M);
qd = zeros(dof, M);
qdd = zeros(dof, M);

% 按关节拆分参数
for joint = 1:dof
    base_idx = (joint-1)*(2*N+1);

    alpha = x(base_idx + (1:N));            % alpha系数，sin项
    beta =  x(base_idx + (N+1 : 2*N));       % beta系数，cos项
    delta = x(base_idx + 2*N + 1);          % 偏移量

    for k = 1:M
        time = t(k);

        % 角度 q
        q(joint,k) = delta;
        % 速度 qd
        qd(joint,k) = 0;   
        % 加速度 qdd
        qdd(joint,k) = 0;
        
        for order = 1:N
            q(joint,k) = q(joint,k) + ...
                (alpha(order)/(order*wf))*sin(order*wf*time) - ...
                (beta(order)/(order*wf))*cos(order*wf*time);

            qd(joint,k) = qd(joint,k) + ...
                alpha(order)*cos(order*wf*time) + ...
                beta(order)*sin(order*wf*time);
            qdd(joint,k) = qdd(joint,k) + ...
                wf * order * (beta(order)*cos(order*wf*time) - ...
                alpha(order)*sin(order*wf*time));
        end

    end
end

end
