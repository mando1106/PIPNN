function [Coefficient_ExTra] = Tradeoff_Modify( Coefficient_ExTra,wf,q_max,q_min,dq_max,ddq_max )

Tf = 2*pi/wf;

t = 0:0.01:Tf;

%%%%%%%%%%%%%%%%%%%%
ddq_ini = Coefficient_ExTra(2)*(wf) + Coefficient_ExTra(4)*(2*(wf)) + Coefficient_ExTra(6)*(3*(wf)) + Coefficient_ExTra(8)*(4*(wf)); 
dq_ini = Coefficient_ExTra(1) + Coefficient_ExTra(3) + Coefficient_ExTra(5) + Coefficient_ExTra(7);
%%%%%%%%%%%%%%%%%%%%

Coefficient_ExTra(1,10:11) = [dq_ini, ddq_ini];

[q,dq,ddq] = Exciting_Trajectory(Coefficient_ExTra,t,wf);

Qmax = max(q);
Qmin = min(q);
dQmax = max(abs(dq));
ddQmax = max(abs(ddq));

Gama_2 = dq_max/dQmax;
Gama_3 = ddq_max/ddQmax;

q_middle= (q_max + q_min)/2;  
Dq_limit = q_max - q_min;

Dq_current = Qmax - Qmin;
q_center = (Qmax + Qmin)/2;

Gama_1 = Dq_limit/Dq_current;

Gama = min([Gama_1,Gama_2,Gama_3]);

Coefficient_ExTra = Coefficient_ExTra*Gama;
q_ini = Coefficient_ExTra(9);

q_center = q_center*Gama;

q_ini = q_ini - (q_center - q_middle);

Coefficient_ExTra(1,9) = q_ini;


end

