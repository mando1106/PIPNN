

thetalist = 0.3*ones(6,1);
dthetalist = 0.6*ones(6,1);
ddthetalist = 0.7*ones(6,1);

pi_min = ones(36,1);
tau_min = min_regressor(thetalist, dthetalist, ddthetalist)*pi_min
tau_min_ = f_W_min(thetalist, dthetalist, ddthetalist)*pi_min

%%  使用matlab 生成的速度会快很多

% 定义输入
thetalist = rand(6,1);
dthetalist = rand(6,1);
ddthetalist = rand(6,1);
pi_min = rand(36,1);

% 创建函数句柄

f2 = @() f_W_min(thetalist, dthetalist, ddthetalist) * pi_min;
f1 = @() min_regressor(thetalist, dthetalist, ddthetalist) * pi_min;


% 测速
t2 = timeit(f2);
t1 = timeit(f1);


fprintf('f_W_min 耗时:     %.6f 秒\n', t2);
fprintf('min_regressor 耗时: %.6f 秒\n', t1);

fprintf('加速比 = %.2fx\n', t1 / t2);
