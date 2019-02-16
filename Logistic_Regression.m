function [Jnew_1,theta]=Logistic_Regession(H,y,X,theta)
alpha=0.001; %learning rate
l=0.008;   % regularization coeffecient
m=length(H);

n=1;
flag=false;
Jnew_1(n)=1/(2*m)*(sum(-y'*log(H'))-sum((1-y')*log(1-H')))+l/(2*m)*sum(theta.^2); %initial cost function
while flag==false
    for k=1:length(theta)  %gradient descent
        temp=(1-((alpha*l)/m))*theta(k);
        theta(k)=temp-((alpha*1/m)*((H-y')*X(k,:)'));
    end
    Hnew=1./(1+exp(-(theta*X))); %updating hypothesis
    H=Hnew;    
    n=n+1;
    Jnew_1(n)=1/(2*m)*(sum(-y'*log(H'))-sum((1-y')*log(1-H')))+l/(2*m)*sum(theta.^2); %new cost function
    if Jnew_1(n-1)-Jnew_1(n)<0 %check if error is increasing
        break 
    end
    q=(Jnew_1(n-1)-Jnew_1(n))./Jnew_1(n-1); % check convergence condition
    if q <.00001
        flag=true;
    end
    
end
%Here is the plotting step
figure();
plot(Jnew_1)
end


