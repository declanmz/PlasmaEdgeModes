clear all
clc

%%%%%%%%%% CST Dispersion
% load('eps_4.txt');  
% load('eps_4.3.txt');
% load('eps_4.6.txt');
% 
% x1=eps_4(:,1);y1=eps_4(:,2); %X1=[-x1(end:-1:1);x1];Y1=[y1(end:-1:1);y1(1:1:end)];
% x2=eps_4_3(:,1);y2=eps_4_3(:,2); %X2=[-x2(end:-1:1);x2];Y2=[y2(end:-1:1);y2(1:1:end)];
% x3=eps_4_6(:,1);y3=eps_4_6(:,2); %X3=[-x3(end:-1:1);x3];Y3=[y3(end:-1:1);y3(1:1:end)];
% 
% plot(x1/180,y1,x2/180,y2,x3/180,y3)
% %plot(X1,Y1,X2,Y2,X3,Y3)
% 
% xlabel('$kd/\pi$','interpreter','latex')
% ylabel('$f$ [GHz]','interpreter','latex')
% 
% legend('$\varepsilon_s=4$','$\varepsilon_s=4.3$','$\varepsilon_s=4.6$','interpreter','latex')

%%%%%%%%%%%%%%% Scaled Dispersion
c=3e8;
d=6.25*10^-3;
w_p=2*pi*3.5e9; %2*pi*7.50e9
f_p=w_p/2/pi;
k_p=w_p/c;

load('eps_4.6.txt');
kd_pi=eps_4_6(:,1)/180;x=kd_pi/d*pi/k_p;
f=eps_4_6(:,2);y=2*pi*f*10^9/w_p;

X=[-x(end:-1:1);x];Y=[y(end:-1:1);y(1:1:end)];

plot(X,Y)

xlabel('$k/k_p$','interpreter','latex')
ylabel('$\omega/\omega_p$ [GHz]','interpreter','latex')

%legend('$\varepsilon_s=4$','$\varepsilon_s=4.3$','$\varepsilon_s=4.6$','interpreter','latex')
