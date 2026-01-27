clc;
clear;

tic;

dof = 6;        % 6个关节
N = 4;          % 3阶Fourier
t_end = 25;     % 10秒
f  = 1/t_end;   % 0.1   #### 如果需要收敛且满足约束  需要改变这个频率
% dt = 0.1;     % 采样步长0.1秒
N_points = 1000; % 为了计算效率 每次采样1000个点 进行计算
dt = t_end / (N_points);
 

       
% optimizer = 'interior-point';
optimizer = 'sqp';

options = optimoptions(@fmincon, ...
    'Algorithm', optimizer, ...
    'MaxIterations', 500, ...
    'MaxFunctionEvaluations', 1e4, ...
    'ConstraintTolerance', 1e-3, ...
    'Display', 'none');
% options = optimoptions(@fmincon, ...
%     'MaxIterations', 1e3, ...
%     'MaxFunctionEvaluations', 1e5, ...
%     'ConstraintTolerance', 1e-3);

% -----------------------------------------------------------------------
% ConstraintTolerance  代表约束允许的范围 MaxIterations  
% MaxIterations  MaxFunctionEvaluations  计算收敛步长
% -----------------------------------------------------------------------


max_trials = 5; % 尝试次数，建议多于20
topK = 30;       % 保留的最优解数量
results = struct( ...
    'x', {}, ...
    'fval', {}, ...
    'exitflag', {}, ...
    'constr_violation', {}, ...
    'cond_num', {}, ...
    'time', {}); 


for trial = 1:max_trials
    % 随机参数生成示例：每个关节有 2*N+1 个参数
    x0 = rand(dof * (2*N + 1), 1);
    %  ----------------------------------------------------------------------
    % Otimization
    % -----------------------------------------------------------------------
    t_trial = tic;   % ⭐ 单次优化开始计时
    try
    [x, fval, exitflag, output] = fmincon( ...
        @(x) Cond_W(x, dof, N, t_end, dt, f), ...   % 匿名函数形式
        x0, ...
        [], [], [], [], [], [], ...
        @(x) Nonlcon(x, dof, N, t_end, dt, f), ...  % 非线性约束
        options);
    catch
        fprintf('第 %d 次优化异常，跳过\n', trial);
        continue;
    end
        % 保存结果
    trial_time = toc(t_trial);   % ⭐ 单次优化耗时

    % 检查约束违反度，fmincon一般output.constrviolation
    constr_violation = output.constrviolation; 
    cond_num = Cond_W(x, dof, N, t_end, dt, f);
    results(end+1).x = x; %#ok<SAGROW>
    results(end).fval = fval;
    results(end).exitflag = exitflag;
    results(end).constr_violation = constr_violation;
    results(end).cond_num = cond_num ;
    results(end).time = trial_time;   % ⭐ 保存时间

    progress = trial / max_trials * 100;

    fprintf('✅ 进度: %.1f%% | Trial %d | fval = %.2f | time = %.2fs\n', ...
        progress, trial, fval, trial_time);


end

%% 过滤满足约束的结果 并记录训练过程结果


% 过滤满足约束的结果
feasible = arrayfun(@(r) r.constr_violation <= options.ConstraintTolerance && r.exitflag > 0, results);

feasible_results = results(feasible);

total_trials = length(results);
feasible_trials = length(feasible_results);
feasible_rate = feasible_trials / total_trials;

% 按 fval 排序，升序（越小越好）
[~, idx] = sort([feasible_results.fval]);

% 取前 topK 条
top_results = feasible_results(idx);
if length(top_results) > topK
    top_results = top_results(1:topK);
end
%% ================== Data folder setting ==================
data_dir = fullfile(pwd, 'data_exp3');   % ← 你只需要改这里
% 例如：
% data_dir = fullfile(pwd, 'data_sqp');
% data_dir = fullfile(pwd, datestr(now,'yyyymmdd_HHMMSS'));

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf('📁 创建数据目录: %s\n', data_dir);
end
%% =========================================================

% 保存最优轨迹参数到 txt 文件
fid = fopen(fullfile(data_dir, 'best_trajectories.txt'), 'w');
for i = 1:length(top_results)
    % fprintf(fid, 'Trajectory %d: cond=%.6f\n', i, top_results(i).fval);
    fprintf(fid, ...
    'Trajectory %d: cond=%.6f | time=%.3f s\n', ...
    i, top_results(i).fval, top_results(i).time);
    fprintf(fid, '%g ', top_results(i).x);
    fprintf(fid, '\n\n');
end
fclose(fid);

total_time = toc;  % 训练全部结束后调用，得到总时间（秒）
fprintf('总训练次数: %d\n', total_trials);
fprintf('合格次数: %d\n', feasible_trials);
fprintf('合格率: %.2f%%\n', feasible_rate * 100);
fprintf('总训练时间: %.1f 秒%d\n', total_time);

% 保存每次优化的条件数和约束满足情况到另一个 txt 文件
fid = fopen(fullfile(data_dir,'optimization_log.txt'), 'wt');

fprintf(fid, '总训练次数: %d\n', total_trials);
fprintf(fid, '合格次数: %d\n', feasible_trials);
fprintf(fid, '合格率: %.2f%%\n\n', feasible_rate * 100);
fprintf(fid, '总训练时间: %.1f 秒\n\n', total_time);
        

fprintf(fid, 'Trial\tconda\tconstr_violation\texitflag\n');
for i = 1:length(results)
    fprintf(fid, '%d\t%.6f\t%.6f\t%d\n', ...
        i, results(i).fval, results(i).constr_violation, results(i).exitflag);
end
fclose(fid);
%% 保存完整优化结果（用于复现实验）
save(fullfile(data_dir, 'all_results.mat'), ...
     'results', ...
     'top_results', ...
     'options', ...
     'optimizer', ...
     'dof', 'N', 't_end', 'dt', 'f');

fprintf('保存完毕。\n');

