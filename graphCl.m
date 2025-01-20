function [cl,cdp]=graphCl(p,Npanel,uinf,step,c)
%%%%%%%%%%%%%%%%%%%%%%%%
%this function graphs the airfoil coeffecient of lift vs. the angle of
%attack
%P          airfoil PARSEC parameters
%Npanel     number of panel to solve for Cl
%uinf       stream velocity magnitude
%step       step between the angles
%c          graph color
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
angle=-30:step:30;   %angles calculation 
cl=[];
cdp=[];
for i=1:length(angle)
    [cl1,~]=solver(p,uinf,angle(i)*pi/180,Npanel);      %Calculating the coeffecient of lift
    cl=[cl,cl1];
end
%Cl plot
line(angle,cl,'color',c,'Marker','o','lineWidth',1.5)
axis([-60 60 -2.5 2.5])
hold on
end