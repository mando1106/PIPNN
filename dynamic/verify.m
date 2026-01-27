% 验证测试  惯性参数集和最小惯性参数及的正确性
% 使用了  Robot Peter 工具箱

clc;clear
tic
thetalist = [0;pi/3;pi/3;-pi/6;pi/2;pi/2];
dthetalist = [pi/2;pi/2;pi/6;0;pi/2;pi/4];
ddthetalist = [pi/2;pi/2;pi/6;pi/6;pi/2;pi/4];
mass_list = [2.920; 6.787; 2.450; 1.707; 1.707; 0.176]; 
%连杆质心相对于坐标系{i}坐标(m)
mass_center_list = [0.0316 -3.1464 -13.8983;
                    131.5620 -0.0210 112.1840;
                    190.3840 0.0410 17.1800;
                    0.0886 21.0083 -2.5014;
                    -0.0886 -21.0083 -2.5014;
                    0 0 8.0000]*10^-3;
mass_center_list = mass_center_list';
%inertia_tensor_list 连杆相对于质心坐标系的惯量张量(kg*m^2)
inertia_tensor_list = zeros(3,3,6);
inertia_tensor_list(:,:,1)  = [42.614 0.046 0.062; 0.046 41.164 -1.386; 0.062 -1.386 31.883]*10^-4;
inertia_tensor_list(:,:,2)  = [100.7 -1.8 1.6; -1.8 1100.8 0; 1.6 0 1087.1]*10^-4;
inertia_tensor_list(:,:,3)  = [31.45 0.48 7.23; 0.48 172.41 -0.15; 7.23 -0.15 166.82]*10^-4;
inertia_tensor_list(:,:,4)  = [20.92 -0.061 0.078; -0.061 16.808 0.992; 0.078 0.992 19.75]*10^-4;
inertia_tensor_list(:,:,5)  = [20.92 -0.061 -0.078; -0.061 16.808 -0.992; -0.078 -0.992 19.75]*10^-4;
inertia_tensor_list(:,:,6)  = [0.9296 0 0; 0 0.9485 0; 0 0 1.5925]*10^-4; 

MDH;

tau_tool = RobotTool_Verify(thetalist,dthetalist,ddthetalist,dh,mass_list',mass_center_list,inertia_tensor_list);
tau_tool = double(tau_tool);

pi_tau=zeros(60,1);
for i=1:6
    pi_tau(1+10*(i-1))=mass_list(i);
    pi_tau((2+10*(i-1)):(4+10*(i-1)))=mass_list(i)*mass_center_list(:,i);
    

    Io=inertia_tensor_list(:,:,i)+ mass_list(i)*(...
        mass_center_list(:,i)'*mass_center_list(:,i)*eye(3)-...
        mass_center_list(:,i)*mass_center_list(:,i)');
    pi_tau((5+10*(i-1)):(10+10*(i-1)))=[Io(1,1),Io(1,2),Io(1,3),Io(2,2),Io(2,3),Io(3,3)]';   
end

addpath('.\utils')

Y=regressor(thetalist, dthetalist, ddthetalist);
tau_re=Y*pi_tau;

%% W * $\theta$  = W_min * $\theta_min$

% W=regressor(thetalist, dthetalist, ddthetalist);
% W_min=min_regressor(thetalist, dthetalist, ddthetalist);
% pi_min= W_min \ (W * pi_tau);

pnum_sum = 60;
pnum_sum_min = 36;
WW = zeros(pnum_sum * 6, pnum_sum);
WW_min = zeros(pnum_sum_min * 6, pnum_sum_min );

pnum_sum_f = 72 ;
pnum_sum_min_f = 48;
WW_f = zeros(pnum_sum_f * 6, pnum_sum_f);
WW_min_f = zeros(pnum_sum_min_f * 6, pnum_sum_min_f );

for i = 1:pnum_sum
    q = unifrnd(-pi, pi, 1, 6);	
    qd = unifrnd(-5*pi, 5*pi, 1, 6);
    qdd = unifrnd(-10*pi, 10*pi, 1, 6);
    
	row1 = 1+6*(i-1); row2 = 6+6*(i-1);
    WW(row1:row2, :) = regressor(q', qd', qdd');
    WW_min(row1:row2, :) = min_regressor(q', qd', qdd');
end

for i = 1:pnum_sum_f
    q = unifrnd(-pi, pi, 1, 6);	
    qd = unifrnd(-5*pi, 5*pi, 1, 6);
    qdd = unifrnd(-10*pi, 10*pi, 1, 6);
    
	row1 = 1+6*(i-1); row2 = 6+6*(i-1);
    WW_f(row1:row2, :) = regressor_f(q', qd', qdd');
    WW_min_f(row1:row2, :) = min_regressor_f(q', qd', qdd');
end

f_x = 0.1 + (0:11)*0.05;    % 随便假设的 每个关节摩擦系数  只需要不一样即可
pi_tau_f = [pi_tau ; f_x']; % 堆成一列
pi_min= WW_min \ (WW * pi_tau);
pi_min_f= WW_min_f \ (WW_f * pi_tau_f);
%% 

thetalist = 0.3*ones(6,1);
dthetalist = 0.6*ones(6,1);
ddthetalist = 0.7*ones(6,1);


%  只有机械臂动力学  不考虑摩擦力
tau  = regressor(thetalist, dthetalist, ddthetalist)*pi_tau
tau_min = min_regressor(thetalist, dthetalist, ddthetalist)*pi_min

%  考虑摩擦力
tau_f  = regressor_f(thetalist, dthetalist, ddthetalist)*pi_tau_f
tau_min_f = min_regressor_f(thetalist, dthetalist, ddthetalist)*pi_min_f

% 任意位置、速度、加速度 算出来 机械臂动力学一样  完成最小化参数集  参数化分解