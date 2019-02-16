clc
clear
close all
load('heartDD.mat');

l=0.008;%regularization coeffoecient
Scaled=zeros(size(b)); %normailzed features
for i=1:size(b,2)   
Scaled(:,i)=b{:,i}/max(b{:,i});        
end
%dividing data into training and testing sets.
Train=Scaled(1:175,1:end-1); %70 percent of complete data
Train_y=Scaled(1:175,end);
Test=Scaled(176:end,1:end-1);% 30 percent of complete data
Test_y=Scaled(176:end,end);
%---------------------------------------------------
x1=(Train(:,1));
x2=(Train(:,2));
x3=(Train(:,3));
x4=(Train(:,4));
x5=(Train(:,5));
x6=(Train(:,6));
x7=(Train(:,7));
x8=(Train(:,8));
x9=(Train(:,9));
x10=(Train(:,10));
x11=(Train(:,11));
x12=(Train(:,12));
x13=(Train(:,13));

%---------------------------------------------------------------
x1T=(Test(:,1));
x2T=(Test(:,2));
x3T=(Test(:,3));
x4T=(Test(:,4));
x5T=(Test(:,5));
x6T=(Test(:,6));
x7T=(Test(:,7));
x8T=(Test(:,8));
x9T=(Test(:,9));
x10T=(Test(:,10));
x11T=(Test(:,11));
x12T=(Test(:,12));
x13T=(Test(:,13));
%Train--------------------------------------------------------
X1=[ones(1,length(Train));(x1.^2)';(x2.^2)';(x3.^2)']; %set of all selected features of all training data
theta1=zeros(1,4); %theta parameters
H1=1./(1+exp(-(theta1*X1))); %forming the hypothesis
[JJ1 theta1]=Regression(H1,Train_y,X1,theta1);
%TEST--------------------------------------------------------
Xtest1=[ones(1,length(Test));(x1T.^2)';(x2T.^2)';(x3T.^2)'];
H_T1=1./(1+exp(-(theta1*Xtest1))); 
J_test1=1/(2*length(H_T1))*sum(-Test_y'*log(H_T1')-(1-Test_y')*log(1-H_T1'))+l/(2*length(H_T1))*sum(theta1(2:length(theta1)).^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
X2=[ones(1,length(Train));exp(x1)';exp(x2)';exp(x3)';exp(x4)';exp(x5)']; %set of all selected features of all training data
theta2=zeros(1,6); %theta parameters
H2=1./(1+exp(-(theta2*X2))); %forming the hypothesis
[JJ2 theta2]=Regression(H2,Train_y,X2,theta2);
%TEST----------------------------------------------------------
Xtest2=[ones(1,length(Test));exp(x1T)';exp(x2T)';exp(x3T)';exp(x4T)';exp(x5T)'];
H_T2=1./(1+exp(-(theta2*Xtest2))); 
J_test2=1/(2*length(H_T2))*sum(-Test_y'*log(H_T2')-(1-Test_y')*log(1-H_T2'))+l/(2*length(H_T2))*sum(theta2(2:length(theta2)).^2); 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
X3=[ones(1,length(Train));(x6.^2)';(x7.^2)';(x8.^2)';(x9.^2)';(x10.^2)']; %set of all selected features of all training data
theta3=zeros(1,6); %theta parameters
H3=1./(1+exp(-(theta3*X3))); %forming the hypothesis
[JJ3 theta3]=Regression(H3,Train_y,X3,theta3);
% %TEST----------------------------------------------------------
Xtest3=[ones(1,length(Test));(x6T.^2)';(x7T.^2)';(x8T.^2)';(x9T.^2)';(x10T.^2)']; %set of all selected features of all training data
H_T3=1./(1+exp(-(theta3*Xtest3))); 
J_test3=1/(2*length(H_T3))*sum(-Test_y'*log(H_T3')-(1-Test_y')*log(1-H_T3'))+l/(2*length(H_T3))*sum(theta3(2:length(theta3)).^2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
 X4=[ones(1,length(Train));exp(x11)';exp(x12)';exp(x13)']; %set of all selected features of all training data
theta4=zeros(1,4); %theta parameters
H4=1./(1+exp(-(theta4*X4))); %forming the hypothesis
[JJ4 theta4]=Regression(H4,Train_y,X4,theta4);
%TEST----------------------------------------------------------
Xtest4=[ones(1,length(Test));exp(x11T)';exp(x12T)';exp(x13T)'];
H_T4=1./(1+exp(-(theta4*Xtest4))); 
J_test4=1/(2*length(H_T4))*sum(-Test_y'*log(H_T4')-(1-Test_y')*log(1-H_T4'))+l/(2*length(H_T4))*sum(theta4(2:length(theta4)).^2); 



