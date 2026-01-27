clc;clear;


thetalist = 0.3*ones(6,1);
dthetalist = 0.6*ones(6,1);
ddthetalist = 0.7*ones(6,1);

pi_min_f = 0.1 + (0:47) * 0.05;
min_regressor_f(thetalist, dthetalist, ddthetalist)
tau_min_f = min_regressor_f(thetalist, dthetalist, ddthetalist)*pi_min_f'