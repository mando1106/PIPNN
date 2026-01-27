function y = Cond_W(x, dof, N, t_end, dt,f)
% generate_trajectory 基于Fourier级数生成机械臂轨迹（角度、速度、加速度）
% min_regressor_f  这个函数是在dynamic 中生成的
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

W = zeros(M * dof, 48);  % 带摩擦参数的最小参数 矩阵列为 48
% 按时间步长进行计算
for k = 1:M
    time = t(k);
    for joint = 1:dof
        base_idx = (joint-1)*(2*N+1);
        alpha = x(base_idx + (1:N));            % alpha系数，sin项
        beta  = x(base_idx + (N+1 : 2*N));      % beta系数，cos项
        delta = x(base_idx + 2*N + 1);          % 偏移量

        % 初始化
        q(joint,k)  = delta;
        qd(joint,k) = 0;
        qdd(joint,k)= 0;

        % Fourier 累加
        for order = 1:N
            w = order * wf;
            sin_wt = sin(w * time);
            cos_wt = cos(w * time);

            q(joint,k)  = q(joint,k)  + (alpha(order)/(w))*sin_wt - (beta(order)/(w))*cos_wt;
            qd(joint,k) = qd(joint,k) + alpha(order)*cos_wt + beta(order)*sin_wt;
            qdd(joint,k)= qdd(joint,k)+ wf * order * (beta(order)*cos_wt - alpha(order)*sin_wt);
        end
    end

    row1 = 1+(k-1)*6; 
    row2 = 6+(k-1)*6;
    W(row1:row2,:) = min_regressor_f(q(:,k), qd(:,k), qdd(:,k));
end

y = cond(W);


end

