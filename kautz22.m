%function kautz2(x,n)
clc
% clear all
close all
% error=norm(B-Y);
options = optimoptions(@fmincon,'Algorithm','interior-point');
% n=10;
n=2;
n1 = n/2;
xx=[0.1;0.2];
save xx;
% xx=[0.5120;0.6999];
%x=[0.4692;0.606];
% x=[0.5041;0.6661];
%  x=[0.9922 ;0.9922;0.0386; -0.0386];
%  x=[-0.227246370890593;-0.0386436149519475;0.0629546053568212;0.0272746773094672];
%  x=[-0.227246370890593;-0.386436149519475;0.0629546053568212;0.272746773094672;0.150252512908033;0.704162410183152;-0.0144257524179423;0.478474833147605;-0.0451643871052473;0.717943209838858];
% x=[0.8437;0.4035];
%  x=[-0.1885;-1.232];
% load states;
% x=[-0.3877;0.6933];
%  x=[0.29;0.23;0.90;0.89];
% x=[0.1;0.1];
%  SS1=[61.5824;18.7884;6.0117;2.3117;1.2217;0.7912;0.5863;0.4754;0.4139;0.3822];
%  load SS3
% x=[0.504;0.505;0.5060;0.5061;0.5062;0.6661;0.6662;0.6663;0.6664;0.6665];
% save x;
%x=[0.4692;0.4693;0.4694;0.4695;0.4696;0.6060;0.6061;0.6062;0.6063;0.6064];
%x=[0.552;0.551;0.550;0.553;0.554;0.015;0.014;0.013;0.011;0.010];
%x=[0.474054925885222;0.474584869556797;0.473857384928248;0.474049716682524;0.473743626802350;0.142343857170433;-0.518882336580626;-0.490929590169505;0.072911953140282;0.116137059926739];
%x=[-0.31;0.27;0.29;0.23];
% x=[-0.91;0.94;0.01;0.04];
% x=[-0.200000000000000;0.200000000000000;0.010000000000000;1.377662105187267e-04];
% X=[-0.310000000000000;5.045401620858544e-07;0.010000000000000;7.941792892014936e-07]
% first x values
% x=[0.659661133618675;0.3100000000;0.120000000000000;0.070000000000000];

[x,fval]=fmincon(@(x)kautzobj(x),x,[],[],[],[],[],[],@(x)con_fun(x),options);








