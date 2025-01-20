function [Cl,maxThickness]=solver(p,uinf,AOA,Npanel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function solves for the coeffecient of lift of an airfoil using the
%Vortex panel method given an input parameters of:
%P          airfoil PARSEC parameters
%uinf       velocity free stream magnitude
%AOA        airfoil angle of attack
%Npanel     Number of panels to solve for
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%geometry dicretization (Forming panels)
dbeta=pi/Npanel;
beta=0;
Z_u0=[];
Z_d0=[];
x0=[];
k=1;
Uinf=uinf*cos(AOA);
Vinf=uinf*sin(AOA);
a=parsec(p);    %PARSEC coeffecients
for i=0:dbeta:pi
    x0(k)=(1-cos(beta))/2;
    [Z_u0(k) ,Z_d0(k)]=yCoord2(a,x0(k));   %coordinate array
    beta=beta+dbeta;
    k=k+1;
end
maxThickness=max(abs(Z_u0-Z_d0));
Z_d0=(Z_d0(end:-1:1));
X=[x0(end:-1:1) x0(2:end)];
Y=[Z_d0 Z_u0(2:end)];
dx=diff(X);
dz=diff(Y);
alpha=atan2(dz,dx);
x_c = dx/2 + X(1:end-1);   %coallocation points
z_c = dz/2 + Y(1:end-1);   %coallocation points
X2=sqrt(dx.^2 + dz.^2);    %Panel lenght
%% Coeffecient matrix formation
aij=zeros(length(x_c)+1);
bij=zeros(length(x_c));
cos_theta=cos(alpha);
sin_theta=sin(alpha);
for i=1:length(x_c)
    Xp = cos_theta.*(x_c(i) - X(1:end-1)) + sin_theta.*(z_c(i) - Y(1:end-1));
    Yp = -sin_theta.*(x_c(i) - X(1:end-1)) + cos_theta.*(z_c(i) - Y(1:end-1));
    thetaj = atan2(Yp,Xp);
    thetaj1 = atan2(Yp,Xp - X2);
    aij(i,1:end-1) = -1/(2*pi)*(thetaj1 - thetaj);
    theta_1 = atan2((z_c(i)-Y(end)),(x_c(i)-X(end)));
    theta_2 = atan2(z_c(i)-Y(end),x_c(i)-10000000*X(end));
    aij(i,length(x_c)+1) = 1/(2*pi)*(theta_1 - theta_2);
    aij(i,i) = 0.5;
    R12 = Xp.^2 + Yp.^2; 
    R22 = (Xp - X2).^2 + Yp.^2;        
    f = (Xp.*log(R12) - (Xp - X2).*log(R22) + 2*Yp.*(thetaj1 - thetaj));
    bij(i,:) = 1/(4*pi).*f; 
    bij(i,i) = 1/(2*pi)*Xp(i)*log(R12(i));
end
%% Right hand side calculation
nvec=[-sin_theta ;cos_theta];
for i=1:length(nvec)
    RHSb(i)=[Uinf Vinf]*nvec(:,i);
end
%Coeffecient matrix
 A = zeros(length(x_c));
A(:,1) = aij(1:end-1,1) - aij(1:end-1,length(alpha)+1);
A(:,end) = aij(1:end-1,length(alpha)) + aij(1:end-1,length(alpha)+1);
A(:,2:end-1) = aij(1:end-1,2:end-2);
RHS =-bij*RHSb';
gamma=A\RHS;    %vortex distribution
for i=1:length(gamma)
    Ueinf(i)=Uinf*x_c(i)+z_c(i)*Vinf+gamma(i);    %velocity field
end
for i = 1:length(gamma)-1
        
    Ue(i) = 2*(Ueinf(i) - Ueinf(i+1))/(X2(i)+X2(i+1));
    Cp(i) = 1 - Ue(i)^2/((Uinf^2)+(Vinf^2)); %coeffecient of pressure       
    
end
%% aerodynamic forces calculation
fx=[];
fy=[];
mj=[];
for j=1:length(Cp)
    fx(j)=Cp(j)*(Y(j+1)-Y(j));
    fy(j)=Cp(j)*(X(j+1)-X(j));
    mj(j)=-fx(j)*(Y(j+1)+Y(j))/2+fy(j)*(((X(j+1)+X(j))/2)-(1/4));
end
Fx=-sum(fx);
Fy=-sum(fy);
M=sum(mj);
Cl=-sin(AOA)*Fx+cos(AOA)*Fy;  %Coeffecient of lift
Cdp=Fx*cos(AOA)+Fy*sin(AOA);  %Coeffecient of drag due to aerodynamic pressure