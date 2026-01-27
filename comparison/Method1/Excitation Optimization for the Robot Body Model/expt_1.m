clear all;
clc;

Q1_set = [-160 160;-160 160;-160 160];
Q2_set = [-160 -30;-160 -30;-160 -30];
Q3_set = [-20 220;-20 220;-20 220];
Q4_set = [-170 170;-170 170;-170 170];
Q5_set = [-115 115; -115 115; -115 115];
Q6_set = [-170 170;-170 170;-170 170];

% Exciting_Time = 6;
% Sampling_Time = 0.004;
% Max_Init_SampTime = 0.008;
% Max_Interval_SampTime = 0.008;

Exciting_Time = 25;
Sampling_Time = 0.004;
Max_Init_SampTime = 0.008;      % 丢前 2 个点
Max_Interval_SampTime = 0.024;  % 每 6 个点取 1 个

Calculate_Init = ceil(Max_Init_SampTime / Sampling_Time);
Calculate_Interval = floor(Max_Interval_SampTime / Sampling_Time);
Calculate_Num = floor(((Exciting_Time / Sampling_Time + 1) - Calculate_Init)/Calculate_Interval + 1);

DOF = 6;

Tf = Exciting_Time;
wf = 2*pi/Tf;

Population = 15;  

Iteration = 10;   

Num_Design_Variate_OneDof = 11;     
Num_Design_Variate = DOF*Num_Design_Variate_OneDof;     

TYPE = 1;
UIO = [randperm(TYPE); randperm(TYPE); randperm(TYPE); randperm(TYPE); randperm(TYPE); randperm(TYPE)];

profile clear;
profile off;
profile on;

for MU = 1:TYPE
    
    % q_max = [Q1_set(UIO(1,MU),2) Q2_set(UIO(2,MU),2) Q3_set(UIO(3,MU),2) Q4_set(UIO(4,MU),2) Q5_set(UIO(5,MU),2) Q6_set(UIO(6,MU),2)]/180*pi;
    % q_min = [Q1_set(UIO(1,MU),1) Q2_set(UIO(2,MU),1) Q3_set(UIO(3,MU),1) Q4_set(UIO(4,MU),1) Q5_set(UIO(5,MU),1) Q6_set(UIO(6,MU),1)]/180*pi;
    % dq_max = [110 110 230 230 230 230]/180*pi;
    % ddq_max = [380; 380; 780; 780; 780; 780]/180*pi;

    q_max   = [pi, 1/2*pi, 2/3*pi, 2/3*pi, 2/3*pi, pi];
    q_min   = [-pi, -1/2*pi, -2/3*pi, -2/3*pi, -2/3*pi, -pi];
    dq_max  = [0.8, 0.8, 0.8, 1.0, 1.0, 1.5];
    ddq_max = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    Coefficient_ExTra_Current = -3 + 6*rand(Population,Num_Design_Variate);
    
    for i = 1:Population
        for Joint = 1:DOF
            XI = Coefficient_ExTra_Current(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint);
            [XI] = Tradeoff_Modify( XI,wf,q_max(Joint),q_min(Joint),dq_max(Joint),ddq_max(Joint));
            Coefficient_ExTra_Current(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint) = XI;
        end
    end
    
    for i = 1:Population
        Value(i) = Objective_BattleMyself_FullModel( Coefficient_ExTra_Current(i,:),wf,Calculate_Num,Calculate_Interval,Calculate_Init,Sampling_Time,DOF,Num_Design_Variate_OneDof );
    end
    
    [Y_total_kbese_i(1), i_best] = min(Value);
    
    for k = 1:Iteration

        %knowledge transfer
        for j = 1:Num_Design_Variate
            Mean_Result(1,j) = sum(Coefficient_ExTra_Current(:,j))/Population;
        end
        
        TF = round(1 + rand);
        Difference_Mean_i = rand(1,Num_Design_Variate).*(Coefficient_ExTra_Current(i_best,1:Num_Design_Variate) - TF*Mean_Result(1,1:Num_Design_Variate));
        
        for i = 1:Population
            Coefficient_ExTra_Current_New(i,1:Num_Design_Variate) = Coefficient_ExTra_Current(i,1:Num_Design_Variate) + Difference_Mean_i;
            
            for Joint = 1:DOF
                XI = Coefficient_ExTra_Current_New(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint);
                [XI] = Tradeoff_Modify( XI,wf,q_max(Joint),q_min(Joint),dq_max(Joint),ddq_max(Joint));
                Coefficient_ExTra_Current_New(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint) = XI;
            end
        end
        
        %Optimization Selected
        for i = 1:Population
            Value_new(i) = Objective_BattleMyself_FullModel( Coefficient_ExTra_Current_New(i,:),wf,Calculate_Num,Calculate_Interval,Calculate_Init,Sampling_Time,DOF,Num_Design_Variate_OneDof );
            
            if Value_new(i) < Value(i)
                Coefficient_ExTra_Current(i,1:Num_Design_Variate) = Coefficient_ExTra_Current_New(i,1:Num_Design_Variate);
                Value(i) = Value_new(i);
            end
        end
        
        
        %learners increase their knowledge by interacting among themselves
        
        for i = 1:Population
            Rand_Subject(1,1:Num_Design_Variate) = rand(1,Num_Design_Variate);
            if i == 1
                Order = 2:Population;
            elseif i == Population
                Order = 1:(Population-1);
            else
                Order1 = 1:(i-1);
                Order2 = (i+1):Population;
                Order = [Order1 Order2];
            end
            h = Order(ceil(rand*(Population-1)));   %Learner
            
            %Compare i and h
            if Value(i) <= Value(h)
                Coefficient_ExTra_Current_New(i,1:Num_Design_Variate) = Coefficient_ExTra_Current(i,1:Num_Design_Variate) + Rand_Subject.*(Coefficient_ExTra_Current(i,1:Num_Design_Variate) - Coefficient_ExTra_Current(h,1:Num_Design_Variate));
            else
                Coefficient_ExTra_Current_New(i,1:Num_Design_Variate) = Coefficient_ExTra_Current(i,1:Num_Design_Variate) + Rand_Subject.*(Coefficient_ExTra_Current(h,1:Num_Design_Variate) - Coefficient_ExTra_Current(i,1:Num_Design_Variate));
            end
            
            for Joint = 1:DOF
                XI = Coefficient_ExTra_Current_New(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint);
                [XI] = Tradeoff_Modify( XI,wf,q_max(Joint),q_min(Joint),dq_max(Joint),ddq_max(Joint));
                Coefficient_ExTra_Current_New(i,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint) = XI;
            end
            
            Value_new(i) = Objective_BattleMyself_FullModel( Coefficient_ExTra_Current_New(i,:),wf,Calculate_Num,Calculate_Interval,Calculate_Init,Sampling_Time,DOF,Num_Design_Variate_OneDof );
            
            if Value_new(i) < Value(i)
                Coefficient_ExTra_Current(i,1:Num_Design_Variate) = Coefficient_ExTra_Current_New(i,1:Num_Design_Variate);
                Value(i) = Value_new(i);
            end
            
        end
        
        [Y_total_kbese_i(k+1), i_best] = min(Value);
        
    end
    
    Best_Sollution_Cycle(MU,1:Num_Design_Variate) = Coefficient_ExTra_Current(i_best,1:Num_Design_Variate);
    
    Y_total_kbese_i(end)
    
end

profile viewer;
profile off;

figure(MU);
X = 0:1:Iteration;
plot(X,Y_total_kbese_i,'d:b','linewidth',1.5);
xlabel('Iteration');
ylabel('Condition number');
title('My Method');

for Joint = 1:DOF
    Coefficient_ExTra(Joint,1:Num_Design_Variate_OneDof) = Best_Sollution_Cycle(1,(Num_Design_Variate_OneDof*(Joint-1)+1):Num_Design_Variate_OneDof*Joint);  
end

