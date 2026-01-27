% dof = 6;      % 6个关节
% N = 3;        % 3阶Fourier
% t_end = 10;   % 10秒
% dt = 0.1;     % 采样步长0.1秒
% f = 0.1;
% % 随机参数生成示例：每个关节有 2*N+1 个参数
% x = rand(dof * (2*N + 1), 1);
% 
% [t, q, qd, qdd] = traj(x, dof, N, t_end, dt , f);
% figure
% plot(t, qdd);
% xlabel('时间 (秒)');
% ylabel('关节角度 (弧度)');
% legend(arrayfun(@(i) sprintf('关节%d', i), 1:dof, 'UniformOutput', false));
% 
% Cond_W(x, dof, N, t_end, dt, f)

%% min_regressor_f  这个函数是在dynamic 中生成的  直接复制粘贴过来

clc;
clear;

tic;

%  ----------------------------------------------------------------------
% 激励轨迹频率和时间对收敛有较大的影响这里需要保证f=1/t_end 以此来保证最终位置的零点
% -----------------------------------------------------------------------
dof = 6;        % 6个关节
N = 4;          % 3阶Fourier
t_end = 25;     % 10秒
f  = 1/t_end;   % 0.1   #### 如果需要收敛且满足约束  需要改变这个频率
% dt = 0.1;     % 采样步长0.1秒
N_points = 100; % 为了计算效率 每次采样100个点 进行计算
dt = t_end / (N_points);

% 随机参数生成示例：每个关节有 2*N+1 个参数
x0 = rand(dof * (2*N + 1), 1);



% 优化

% options = optimoptions(@fmincon,'MaxIterations',300,'MaxFunctionEvaluations',1000);
% % [x, fval, extiflag, output] = fmincon(@Cond_W(x, dof, N, t_end, dt,f), x0, [], [], [], [], [], [], @Nonlcon(x, dof, N, t_end, dt , f), options);

options = optimoptions(@fmincon, ...
    'MaxIterations', 1e4, ...
    'MaxFunctionEvaluations', 1e6, ...
    'ConstraintTolerance', 1e-2);

% options = optimoptions('fmincon', ...
%     'Algorithm', 'interior-point', ...
%     'MaxIterations', 1e4, ...
%     'MaxFunctionEvaluations', 1e6, ...
%     'Display', 'iter', ...
%     'ConstraintTolerance', 1e-3);  % 放宽约束容差

% 下面的代码是无约束时 优化最小条件数的过程
% [x, fval, exitflag, output] = fmincon( ...
%     @(x) Cond_W(x, dof, N, t_end, dt, f), ...
%     x0, [], [], [], [], [], [], [], options);

%  ----------------------------------------------------------------------
% Otimization
% -----------------------------------------------------------------------
% 添加约束  优化最小条件数
[x, fval, exitflag, output] = fmincon( ...
    @(x) Cond_W(x, dof, N, t_end, dt, f), ...   % 匿名函数形式
    x0, ...
    [], [], [], [], [], [], ...
    @(x) Nonlcon(x, dof, N, t_end, dt, f), ...  % 非线性约束
    options);

%  ----------------------------------------------------------------------
% Otimization  这个优化方法在这里不收敛
% -----------------------------------------------------------------------
% A = []; b = [];
% Aeq = []; beq = [];
% lb = []; ub = [];
%  模式搜索算法
% optns_pttrnSrch = optimoptions('patternsearch');
% optns_pttrnSrch.Display = 'iter';
% optns_pttrnSrch.StepTolerance = 1e-1;
% optns_pttrnSrch.FunctionTolerance = 10;
% optns_pttrnSrch.ConstraintTolerance = 1e-6;
% optns_pttrnSrch.MaxTime = inf;
% optns_pttrnSrch.MaxFunctionEvaluations = 1e+6;
% 
% [x,fval] = patternsearch(@(x) Cond_W(x, dof, N, t_end, dt, f), x0, ...
%                          A, b, Aeq, beq, lb, ub, ...
%                          @(x) Nonlcon(x, dof, N, t_end, dt, f), optns_pttrnSrch);



toc;



%% 测试最终得到的 轨迹和条件数

Cond_W(x, dof, N, t_end, dt, f)

% dof = 6;      % 6个关节
% N = 3;        % 3阶Fourier
% t_end = 10;   % 10秒
% dt = 0.1;     % 采样步长0.1秒
% f = 0.1;

dof = 6;        % 6个关节
N = 4;          % 3阶Fourier
t_end = 25;     % 10秒
f  = 1/t_end;   % 0.1   #### 如果需要收敛且满足约束  需要改变这个频率
% dt = 0.1;     % 采样步长0.1秒
N_points = 100; % 为了计算效率 每次采样100个点 进行计算
dt = t_end / (N_points);

 x = [-0.00125553 0.39786 -0.396604 -2.43067e-09 -0.00221381 0.00708431 -0.00398502 -1.13007e-09 7.00941e-16 0.127236 -0.394315 0.00546931 0.26161 0.00963598 -0.0254531 0.00350521 0.00768864 -1.45546e-16 0.132895 0.242948 -0.386341 0.0104985 -0.0115165 0.0510888 -0.0573366 0.0203372 7.64011e-16 -0.0242389 -0.219225 0.0080026 0.235461 0.0914171 -0.259095 0.0785633 0.0477708 1.60869e-15 -0.0254436 -0.169347 -0.0115477 0.206338 0.131576 -0.327918 -0.00262778 0.133036 1.34066e-16 -0.548465 0.238835 0.308808 0.000821483 0.0196374 -0.0627791 0.0351916 8.64757e-05 -5.19909e-16 
];
[t, q, qd, qdd] = traj(x, dof, N, t_end, dt , f);
figure
plot(t, q);
xlabel('时间 (秒)');
ylabel('关节角度 ');
figure
plot(t, qd);
xlabel('时间 (秒)');
ylabel('关节速度 ');
figure
plot(t, qdd);
xlabel('时间 (秒)');
ylabel('关节加速度 ');
legend(arrayfun(@(i) sprintf('关节%d', i), 1:dof, 'UniformOutput', false));
