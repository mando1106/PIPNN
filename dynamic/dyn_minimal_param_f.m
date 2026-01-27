%% dyn_minimal_param_math.m
% @brief: QR decomposition to figure out minimal parameter set (direct arithmetic expression)
% @param[out] W_min: regression matrix for minimal parameter set 
% @param[out] min_param_ind: minimal parameter set
% @param[out] pnum_min: volume of minimal parameter set
% @param[out] R1: matrix consisting of independent columns
% @param[out] R2: matrix consisting of dependent columns
% @note: Unit:mm(Nmm) is used throughout the project.

clc;clear
addpath('.\utils')
load("K.mat")
%% PARAMETER

% regression matrix for standard parameter set 
W   = evalin('base', 'K');   %(syms 6 x 60)
W_f = evalin('base', 'K_f'); %(syms 6 x 72)

pnum_sum = size(W_f,2);
%% INSTANTIATION
% q = zeros(1, 6);
% qd = zeros(1, 6);
% qdd = zeros(1, 6);
min_param_ind = zeros(1, pnum_sum);		% the index of minimal parameters 
WW = zeros(pnum_sum * 6, pnum_sum);
for i = 1:pnum_sum
    q = unifrnd(-pi, pi, 1, 6);	
    qd = unifrnd(-5*pi, 5*pi, 1, 6);
    qdd = unifrnd(-10*pi, 10*pi, 1, 6);
    
	row1 = 1+6*(i-1); row2 = 6+6*(i-1);
    WW(row1:row2, :) = regressor_f(q', qd', qdd');

    disp(['<INFO> Param No.', num2str(i), ' SUBSTITUTED!!']);
end

%% LOOK FOR NAN AND INF
if (sum(sum(isnan(WW))) > 0)
	pause
else
    disp('No NaN in matrix.');
end

if (sum(sum(isinf(WW))) > 0)
	pause
else
    disp('No Inf in matrix.');
end

%% QR DECOMPOSITION
% WW=Q*R, WW:(6*pnum_sum, pnum_sum), Q:(6*pnum_sum, 6*pnum_sum), R:(6*pnum_sum, pnum_sum)
[Q, R] = qr(WW);
pnum_min = 0;	% number of independent parameter
for i = 1:pnum_sum
   if (abs(R(i, i)) < 10^(-5))
       min_param_ind(i) = 0;
   else
       min_param_ind(i) = 1;
       pnum_min = pnum_min + 1;
   end
end
disp('<INFO> QR DECOMPOSITION complete!!');

W_min_f = sym(zeros(6, pnum_min));	% regression matrix (minimal set)
R1 = zeros(pnum_min, pnum_min);
R2 = zeros(pnum_min, pnum_sum - pnum_min);
cind = 1; cdep = 1;	% count the number of independent and dependent columns
for i = 1:pnum_sum
   if (min_param_ind(i) == 1)
      W_min_f(:, cind) = W_f(:, i);		% compose independent columns in W to form matrix WB
      R1(1:pnum_min, cind) = R(1:pnum_min, i);	% compose independent columns in R to form matrix RB
      cind = cind + 1;
   else
      R2(1:pnum_min, cdep) = R(1:pnum_min, i);
      cdep = cdep + 1;
   end
end
disp('<INFO> WB (W_min) matrix OBTAINED!!');
%% 

fid = fopen('.\utils\W_min_f.txt', 'w');
% 获取矩阵大小
[m, n] = size(W_min_f);
% 先写出初始化部分
fprintf(fid, '%% Initialize W_min_f as zeros\n');
fprintf(fid, 'W_min_f = zeros(%d, %d);\n\n', m, n);

% 写非零元素
for i = 1:m
    for j = 1:n
        val = W_min_f(i, j);
        if val ~= 0
            fprintf(fid, 'W_min_f(%d, %d) = %s;\n', i, j, char(val));
        end
    end
end

% 写出完整矩阵形式
fprintf(fid, '\n%% Final W_min_f matrix\n');
fprintf(fid, 'W_min_f = [\n');
for i = 1:m
    for j = 1:n
        val = W_min_f(i, j);
        if j < n
            fprintf(fid, '%s, ', char(val));
        else
            fprintf(fid, '%s', char(val));
        end
    end
    fprintf(fid, ';\n');
end
fprintf(fid, '];\n');

fclose(fid);

%% RETURN VARIABLE TO BASE WORKSPACE
% assignin('base', 'W_min', W_min_f);
% assignin('base', 'min_param_ind', min_param_ind);
% assignin('base', 'pnum_min', pnum_min);
% assignin('base', 'R1', R1);
% assignin('base', 'R2', R2);

q_sym = sym('q%d',[6,1],'real');
dq_sym = sym('dq%d',[6,1],'real');
ddq_sym = sym('ddq%d',[6,1],'real');
g=sym('g','real');
W_min_ =subs(W_min_f, g, 9.81);

matlabFunction(W_min_, ...
    'File', 'utils/min_regressor_f', ...
    'Vars', {q_sym, dq_sym, ddq_sym}, ...
    'Optimize', true);


rmpath('.\utils')




