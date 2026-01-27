function [ q ,dq ,ddq ] = Exciting_Trajectory( Coefficient_ExTra,t,wf )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

a1 = Coefficient_ExTra(:,1);
b1 = Coefficient_ExTra(:,2);
a2 = Coefficient_ExTra(:,3);
b2 = Coefficient_ExTra(:,4);
a3 = Coefficient_ExTra(:,5);
b3 = Coefficient_ExTra(:,6);
a4 = Coefficient_ExTra(:,7);
b4 = Coefficient_ExTra(:,8);

q_ini = Coefficient_ExTra(:,9);
dq_ini = Coefficient_ExTra(:,10);
ddq_ini = Coefficient_ExTra(:,11);

q = a1/(wf).*sin((wf).*t) - b1/(wf).*cos((wf).*t) + a2/(2.*(wf)).*sin(2.*(wf).*t) - b2/(2.*(wf)).*cos(2.*(wf).*t) + a3/(3.*(wf)).*sin(3.*(wf).*t) - b3/(3.*(wf)).*cos(3.*(wf).*t) + a4/(4.*(wf)).*sin(4.*(wf).*t) - b4/(4.*(wf)).*cos(4.*(wf).*t) + ...
        q_ini - dq_ini/(wf).*sin((wf).*t) + ddq_ini/((wf)^2).*cos((wf).*t);

dq  = a1.*cos((wf).*t) + b1.*sin((wf).*t) + a2.*cos(2.*(wf).*t) + b2.*sin(2.*(wf).*t) + a3.*cos(3.*(wf).*t) + b3.*sin(3.*(wf).*t) + a4.*cos(4.*(wf).*t) + b4.*sin(4.*(wf).*t) - ddq_ini/(wf).*sin((wf).*t) - dq_ini.*cos((wf).*t);

ddq = -a1.*(wf).*sin((wf).*t) + b1.*(wf).*cos((wf).*t) - a2.*(2.*(wf)).*sin(2.*(wf).*t) + b2.*(2.*(wf)).*cos(2.*(wf).*t) - a3.*(3.*(wf)).*sin(3.*(wf).*t) + b3.*(3.*(wf)).*cos(3.*(wf).*t) - a4.*(4.*(wf)).*sin(4.*(wf).*t) + ...
            b4.*(4.*(wf)).*cos(4.*(wf).*t) - ddq_ini.*cos((wf).*t) + dq_ini.*(wf).*sin((wf).*t);
       
end

