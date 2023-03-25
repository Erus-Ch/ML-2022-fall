clear
clc

A = [5 5 0 5; 5 0 3 4;3 4 0 3; 0 0 5 3; 5 4 4 5;5 4 5 5];
[U,S,V] = svd(A);
Vtranspose=V';
A_test = U*S*Vtranspose;
U2=U(1:6,1:2);S2=S(1:2,1:2);V2=Vtranspose(1:2,1:4);
A2= U2*S2*V2;
Bob = [5 5 0 0 0 5]';
Bob_Proj = U2'*Bob;
Bob_2D = pinv(S2)*Bob_Proj;
Bob_2D = Bob_2D';






