function [Jnew_1,theta]=Regession(H,y,X,theta)
alpha=0.001; %learning rate

m=length(H);

n=1;
flag=false;
Jnew_1(n)=1/(2*m)*sum(((H-y').^2)); %initial cost function
while flag==false
    for k=1:length(theta)  %gradient descent 
        temp=theta(k);
        theta(k)=temp-((alpha*1/m)*((H-y')*X(k,:)'));
    end
    Hnew=theta*X; %updating hypothesis
    H=Hnew;
    n=n+1
    Jnew_1(n)=1/(2*m)*sum(((Hnew-y').^2)); %new cost function
    if Jnew_1(n-1)-Jnew_1(n)<0 %checking if error is increasing
        break
    end
    q=(Jnew_1(n-1)-Jnew_1(n))./Jnew_1(n-1); %checking convergence condition
    if q <.001
        flag=true;
    end
    %returning the values of theta to zeros again for the 2nd big iteration
end
%Here is the plotting step

figure();
plot(Jnew_1)
end


