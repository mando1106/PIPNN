

%@input:q:joint angle for every link,  q(6X1).
%@input:dq:joint angle velocity for every link,  dq(6X1).
%@input:ddq:joint angle acculate for every link,  ddq(6X1).
%@input:g:9.81.
%@input:dh stanard for modified DH, dh_list = [alpha; a; d; theta];],  DH(4X6).
%@input:m:the mess of link,  m(1X6).
%@input:Pc sandard the mess center of link,example:Pc(1,1) standsrd for the center of x axis on first link, Pc(2,1)standsrd for the center of y axis on first link,Pc(3,1)standsrd for the center of z axis on first link.  Pc(3X6).
%@input:Ic sandard the inertia tensor of link, example:Ic(:,:,1)standard for the first link inertia tensor, and Ic(:,:,1) is a symmetry matrix, Ic(3X3X6).
%@input:f_tip:external force and torque,f_tip(1,:) stanard for the force, f_tip(2,:) stanard for the torque, f_tip(2X3)

%@output:taulist : the every link need torque,  taulist(6X1)

%R：3x3,旋转矩阵，P：3x6,后一连杆坐标系在前一连杆坐标系中的位置
%w：3x6,连杆角速度，dw：3x6,连杆角加速度，dv：3x6,连杆线加速度，dvc：3x6,连杆质心线加速度
%Ic:3x3x6,等同于inertia_tensor_list
%Pc:3x6, mass_center_list的转置
%F:3x6,各轴所受合力，N:3x6,各轴所受合力矩
%f:3x6,前一轴给后一轴的力，n:3x6,前一轴给后一轴的力矩

clc;clear
tic
q_sym = sym('q%d',[6,1],'real');
dq_sym = sym('dq%d',[6,1],'real');
ddq_sym = sym('ddq%d',[6,1],'real');
m=sym("m%d",[1,6],'real');
Pc=sym("Pc",[3,6],'real');
Ic=sym("Ic",[3,3,6],'real');
g=sym('g','real');


% alpha=[0,     pi/2,  0,           0,         pi/2,      -pi/2];
% a=    [0,     0,     -300,      -276,    0,         0].*0.001;
% d=    [121.5, 0,     0,           110.5,    90,     82].*0.001;
% theta=[0,     -pi/2,  0,           -pi/2,     0,         0];
% dh = [alpha; a; d; theta];

MDH;


dh=sym(dh);

alpha = dh(1,:);
a = dh(2,:);
d = dh(3,:);
theta = dh(4,:);
Z=[0;0;1];
%转换矩阵建立

T=sym(zeros(4,4,6));R=sym(zeros(3,3,6));P=sym(zeros(3,6));
for i = 1:6
    %############### MDH参数 单个转移矩阵表示为
    %############### A=Rot(alpha)*Tran(a)*Tran(d)*Rot(theta)  
    Rot_theta_q=[cos(q_sym(i)) -sin(q_sym(i)) 0 0;sin(q_sym(i)) cos(q_sym(i)) 0 0;0 0 1 0;0 0 0 1];  
    Rot_theta=[cos(theta(i)) -sin(theta(i)) 0 0;sin(theta(i)) cos(theta(i)) 0 0;0 0 1 0;0 0 0 1];
    Tran_d=[1 0 0 0; 0 1 0 0; 0 0 1 d(i); 0 0 0 1];
    Tran_a=[1 0 0 a(i); 0 1 0 0; 0 0 1 0; 0 0 0 1];
    Rot_alpha=[1 0 0 0; 0 cos(alpha(i)) -sin(alpha(i)) 0; 0 sin(alpha(i)) cos(alpha(i)) 0; 0 0 0 1];
    T(:,:,i)=Rot_alpha*Tran_a*Tran_d*Rot_theta*Rot_theta_q; 
    R(:,:,i)=T(1:3,1:3,i);
    P(:,i)=T(1:3,4,i);
end

%运动学正向递推
w0 = zeros(3,1); dw0 = zeros(3,1);

dv0 = [0;0;g];
w = sym(zeros(3,6)); dw =sym(zeros(3,6));
dv = sym(zeros(3,6)); dvc = sym(zeros(3,6));
F = sym(zeros(3,6)); N = sym(zeros(3,6));

%i = 0
w(:,1) = R(:,:,1)' * w0 + dq_sym(1) * Z;
dw(:,1) = R(:,:,1)' * dw0 + cross(R(:,:,1)' * w0, dq_sym(1) * Z) + ddq_sym(1) * Z;
dv(:,1) = R(:,:,1)' * (cross(dw0,P(:,1)) + cross(w0,cross(w0, P(:,1))) + dv0);
dvc(:,1) = cross(dw(:,1), Pc(:,1))+cross(w(:,1), cross(w(:,1), Pc(:,1))) + dv(:,1);
for i = 1:5
   w(:,i+1) = R(:,:,i+1)' * w(:,i) + dq_sym(i+1) * Z ;
   dw(:,i+1) = R(:,:,i+1)' * dw(:,i) + cross(R(:,:,i+1)' * w(:,i), dq_sym(i+1) * Z)+ ddq_sym(i+1) * Z;
   dv(:,i+1) = R(:,:,i+1)' * (cross(dw(:,i), P(:,i+1)) + cross(w(:,i), cross(w(:,i), P(:,i+1))) + dv(:,i));
   dvc(:,i+1) = cross(dw(:,i+1), Pc(:,i+1)) + cross(w(:,i+1), cross(w(:,i+1), Pc(:,i+1))) + dv(:,i+1);
end

for i = 1:6
   F(:,i)=m(i)*dvc(:,i) ;
   N(:,i)=Ic(:,:,i) * dw(:,i) + cross(w(:,i), Ic(:,:,i) * w(:,i));
end

%动力学逆向递推
%先计算杆6的力和力矩
taulist = sym(zeros(6,1));
f=sym(zeros(3,6)); n=sym(zeros(3,6));

f(:,6) = F(:,6) ;
n(:,6) = N(:,6)  + cross(Pc(:,6), F(:,6));
taulist(6) = n(:,6)' * Z;
%再计算杆5到1的力和力矩
for i=5:-1:1
   f(:,i) = R(:,:,i+1) * f(:,i+1) + F(:,i);
   n(:,i) = N(:,i) + R(:,:,i+1) * n(:,i+1) + cross(Pc(:,i), F(:,i))...
            + cross(P(:,i+1), R(:,:,i+1) * f(:,i+1));
   taulist(i) = n(:,i)' * Z;
   
end     
toc
%% 验证准确性
tic
thetalist = [0;pi/3;pi/3;-pi/6;pi/2;pi/2];
dthetalist = [pi/2;pi/2;pi/6;0;pi/2;pi/4];
% ddthetalist = [0;pi/34;0;pi/5;0;0.34];
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

% [torque] = RobotTool_Verify(q, dq,ddq,dh, mass_list , mass_center_list, inertia_tensor_list)
f_tip = zeros(2,3);

tau_tool = RobotTool_Verify(thetalist,dthetalist,ddthetalist,dh,mass_list',mass_center_list,inertia_tensor_list);
tau_tool=double(tau_tool);

v=subs(taulist,Ic,inertia_tensor_list);
v=subs(v,[q_sym dq_sym ddq_sym],[thetalist dthetalist ddthetalist]);
v=subs(v,g,9.81);
v=subs(v,Pc,mass_center_list);
v=subs(v,m,mass_list');
tau_NN=double(v);


toc
%% 

[K,K_f] = cacluate_K(dh,w,dw,dv,R,P,q_sym,dq_sym, ddq_sym);
save('utils/K.mat', 'K', 'K_f');

% zero_columns = find(all( == 0, 1));
pi_tau=zeros(60,1);
for i=1:6
    pi_tau(1+10*(i-1))=mass_list(i);
    pi_tau((2+10*(i-1)):(4+10*(i-1)))=mass_list(i)*mass_center_list(:,i);
    

    Io=inertia_tensor_list(:,:,i)+ mass_list(i)*(...
        mass_center_list(:,i)'*mass_center_list(:,i)*eye(3)-...
        mass_center_list(:,i)*mass_center_list(:,i)');
    pi_tau((5+10*(i-1)):(10+10*(i-1)))=[Io(1,1),Io(1,2),Io(1,3),Io(2,2),Io(2,3),Io(3,3)]';   
end

Y=subs(K,[q_sym dq_sym ddq_sym],[thetalist dthetalist ddthetalist]);
Y=subs(Y,g,9.81);
tau_re=double(Y)*pi_tau;


addpath('.\utils')
KK = subs(K, g, 9.81);
matlabFunction(KK, ...
    'File', 'utils/regressor', ...
    'Vars', {q_sym, dq_sym, ddq_sym}, ...
    'Optimize', true);

KK_f = subs(K_f, g, 9.81);
matlabFunction(KK_f, ...
    'File', 'utils/regressor_f', ...
    'Vars', {q_sym, dq_sym, ddq_sym}, ...
    'Optimize', true);


toc


function [K,K_f] = cacluate_K(dh,w,dw,dv,R,P,q,dq, ddq)

n = 6;

for i = 1:n
    px(:,:,i) = [0,-P(3,i),P(2,i);P(3,i),0,-P(1,i);-P(2,i),P(1,i),0];
    wx(:,:,i) = [0,-w(3,i),w(2,i);w(3,i),0,-w(1,i);-w(2,i),w(1,i),0];
    dwx(:,:,i) = [0,-dw(3,i),dw(2,i);dw(3,i),0,-dw(1,i);-dw(2,i),dw(1,i),0];
    dvx(:,:,i)=[0,-dv(3,i),dv(2,i);dv(3,i),0,-dv(1,i);-dv(2,i),dv(1,i),0];
    Wdot(:,:,i) = [w(:,i)',zeros(1,3);0,w(1,i),0,w(2,i),w(3,i),0;0,0,w(1,i),0,w(2,i),w(3,i)];
    dWdot(:,:,i) = [dw(:,i)',zeros(1,3);0,dw(1,i),0,dw(2,i),dw(3,i),0;0,0,dw(1,i),0,dw(2,i),dw(3,i)];
    DK(:,:,i) = [dv(:,i),dwx(:,:,i)+wx(:,:,i)*wx(:,:,i),zeros(3,6); ...
        zeros(3,1),-dvx(:,:,i),dWdot(:,:,i)+wx(:,:,i)*Wdot(:,:,i)];
end
for i=1:n-1
     RI1(:,:,i) = [R(:,:,i+1),zeros(3,3);px(:,:,i+1)*R(:,:,i+1),R(:,:,i+1)];
end

DKKKK = [];
for i =1:n-1
    D = [];
    RR = eye(6,6);
    for m = i:n-1
        RR = RR * RI1(:,:,m);
        D = [D,RR * DK(:,:,m+1)];
    end
    DKKK = [zeros(6,(i-1)*10),DK(:,:,i),D];
    DKKKK = [DKKKK;DKKK];
end
DKKKK = [DKKKK;zeros(6,(n-1)*10),DK(:,:,n)];
% size(DKKKK)

%取tau值
K = [];
F = [];
for i =1:n
    K = [K;DKKKK(6*i,:)];
end
% size(K)


for i=1:n
    F = [F;zeros(1,2*(i-1)),dq(i),sign(dq(i)),zeros(1,10-2*(i-1))];
%    F = [F;zeros(1,2*(i-1)),dq(i),tanh(dq(i)/0.001),zeros(1,10-2*(i-1))];
%     F = [F;zeros(1,3*(i-1)),dq(i),sign(dq(i)),ddq(i),zeros(1,15-3*(i-1))];
%     F = [F;zeros(1,3*(i-1)),dq(i),tanh(dq(i)/0.05),ddq(i),zeros(1,15-3*(i-1))];

%     F = [F;zeros(1,3*(i-1)),1,1,1,zeros(1,15-3*(i-1))];
end
K_f = [K,F];


end




