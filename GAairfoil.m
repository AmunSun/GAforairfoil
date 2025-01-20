function [cloriginal,clfittest,fittest]=GAairfoil(genNo,p0,range,uinf,AOA,Npanel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function optimizes an airfoil shape based on the coeffecient of lift
%value
%genNo      number of generations to mate
%p0         Original airfoil to oprimize
%range      Randomizer range to vary the PARSEC parameters
%uinf       flow free stream velocity
%AOA        airfoil angle of attack
%Npanel     number of panels for the fitness function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%genetic parameters
[cloriginal,~]=solver(p0,uinf,AOA,Npanel);
popsize=48;  %population size
transprob=0.05;  %transcendence percentage 
crossprob=0.75;    %cross over percentage
mutprob=0.2;       %mutation percentage
newpop=[];
for k=1:genNo
cl=[];
p=[];
%population evaluation (starting from the second generation)
for i=1:length(newpop)
    p1=newpop(i,:);
    [clnew, ~]=solver(p1,uinf,AOA,Npanel);  %fitness evaluation
    cl=[cl;clnew];
    p=[p;p1];
end
%first population initialization
for i=1:popsize-length(newpop)
    p1=randp(p0,range);
    [clnew, maxThickness]=solver(p1,uinf,AOA,Npanel); %fitness evaluation
     %geometric constrain
    if maxThickness>0.1 
        clnew=cloriginal;
    elseif maxThickness<0.01  
        clnew=cloriginal;
    end
    cl=[cl;clnew];
    p=[p;p1];
end
pop=p;
%constraining the coeffecient of lift
for i=1:length(cl)
if cl(i)<=cloriginal
        cl(i)=cloriginal;
end
end
%sorting the individuals by the fittest
fi=cl./sum(cl);
[fittest,ind]=sort(fi,'descend');
fittest=fittest(1:ceil(transprob*popsize));
ind=ind(1:ceil(transprob*popsize));
if k~=genNo
    newpop=pop(ind,:);
    %crossover
    for i=1:ceil(crossprob*popsize)
        indv1=randi([1,popsize],1);
        indv2=randi([1,popsize],1);
        crossindex=randi([1,11],1);
       newpop=[newpop;pop(indv1,1:crossindex) pop(indv2,crossindex+1:end)];
    end
    %mutation
    for i=1:ceil(mutprob*popsize)
        indv=pop(randi([1,popsize],1),:);
        mutindex=randi([1,11],1);
        pmut=randp(p0,range);
        indv(mutindex)=pmut(mutindex);
        newpop=[newpop;indv];
    end
end
end
%choosing the tournemnt winner or the most evolved individual
fittest=pop(ind(1),:);
clfittest=cl(ind(1));
if clfittest==cloriginal
    fittest=p0;
end
%plotting the original airfoil vs. the evolved (optimized)
plotairfoil(fittest,'k')
axis('equal')
hold on
plotairfoil(p0,'r')
legend('Optimized','original')
xlabel('X/C')
ylabel('Y/C')
title('Airfoil shape')
end