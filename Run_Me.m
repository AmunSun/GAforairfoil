%initial individual
% p0=[0.0155 0.296632 0.060015 -0.4515 0.296632 -0.06055 0.453 0 0.001260 0 7.36]; %%NACA 0012
p0=[0.0216 0.3445 0.07912 -0.6448 0.17 -0.033797 0.6748 0 0 0 0];  %%NACA 2411
range=[0.0015 0.025 0.015 -0.01 0.02 -0.015 0.075 0 0 -0.175 0.05];
%range=[0.02 0.023 0.32 0.37 0.077 0.08 -0.63 -0.65 0.15 0.19 -0.02 -0.05 0.6 0.75 0 0 0 0 -4.55 -4.9 15 15.1];
%Solver parameters
Npanel=200;
uinf=1;
AOA=5*pi/180;
genNo=20;       %number of generations
%Genetic solution
[cloriginal,clfittest,fittest]=GAairfoil(genNo,p0,range,uinf,AOA,Npanel);
%ploting and graphing
fprintf(' Original   Cl= %f \n Optimized  Cl= %f \n',cloriginal,clfittest)
figure
graphCl(fittest,Npanel,uinf,5,'k');
graphCl(p0,Npanel,uinf,5,'r');
legend('Optimized','original')
xlabel('AOA')
ylabel('Cl')
line([-100 100],[0,0],'color','k','LineWidth',1)
line([0,0],[-10 10],'color','k','LineWidth',1)
title('Coeffecient of lift vs. Angle of attack')
grid on