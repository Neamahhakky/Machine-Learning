clc
clear
close all
param=tabularTextDatastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
param=read(param);
Data=[];
%converting the columns with character values to numeric values.
for i=3:size(param,2)
    if(i==8||i==17)
        temp=str2num(char(param{:,i}));
        Data=[Data temp];
    else
        temp=param{:,i};
        Data=[Data temp];
    end
end
%normalizing the data
Scaled=zeros(size(Data));
for i=1:size(Data,2)   
Scaled(:,i)=Data(:,i)/max(Data(:,i));        
end
%dividing the data into training and testing sets.
Train=Scaled(1:15130,2:end); %70 percent of complete data
Train_y=Scaled(1:15130,1);
Test=Scaled(15131:end,2:end);% 30 percent of complete data
Test_y=Scaled(15131:end,1);
%---------------------------------------------------
%training data from row 2 to 15130  
x1=(Train(:,3));
x2=(Train(:,4));
x3=(Train(:,6));
x4=(Train(:,10));
x5=(Train(:,11));
x6=(Train(:,12));
x7=(Train(:,16));
x8=(Train(:,18));
%---------------------------------------------------------------
%test data from 15131 to 21614 
x1T=Test(:,3);
x2T=Test(:,4);
x3T=Test(:,6);
x4T=Test(:,10);
x5T=Test(:,11);
x6T=Test(:,12);
x7T=Test(:,16);
x8T=Test(:,18);
%Train--------------------------------------------------------
X1=[ones(1,length(Train));x2';(x3.^2)';x5']; %set of all selected features of all training data
theta1=zeros(1,4); %theta parameters
H1=theta1*X1; %forming the hypothesis
[JJ1 theta1]=Regression(H1,Train_y,X1,theta1);
%TEST--------------------------------------------------------
Xtest1=[ones(1,length(Test));x2T';(x3T.^2)';x5T'];
H_T1=theta1*Xtest1; 
J_test1=(1/(2*length(H_T1)))*sum((H_T1-Test_y').^2);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
X2=[ones(1,length(Train));exp(x4)';exp(-x6)';exp(x7)']; %set of all selected features of all training data
theta2=zeros(1,4); %theta parameters
H2=theta2*X2; %forming the hypothesis
[JJ2 theta2]=Regression(H2,Train_y,X2,theta2);
%TEST----------------------------------------------------------
Xtest2=[ones(1,length(Test));exp(x4T)';exp(-x6T)';exp(x7T)'];
H_T2=theta2*Xtest2; 
J_test2=(1/(2*length(H_T2)))*sum((H_T2-Test_y').^2);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
X3=[ones(1,length(Train));(x2.^(1/2))';(x3.^2)';exp(x4)';x5';exp(-x6)']; %set of all selected features of all training data
theta3=zeros(1,6); %theta parameters
H3=theta3*X3; %forming the hypothesis
[JJ3 theta3]=Regression(H3,Train_y,X3,theta3);
%TEST----------------------------------------------------------
Xtest3=[ones(1,length(Test));(x2T.^(1/2))';(x3T.^2)';exp(x4T)';x5T';exp(-x6T)'];
H_T3=theta3*Xtest3; 
J_test3=(1/(2*length(H_T3)))*sum((H_T3-Test_y').^2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train---------------------------------------------------------
 X4=[ones(1,length(Train));(x1.^(1/2))';(x2.^(1/2))';(x3.^2)';(x8.^(1/2))']; %set of all selected features of all training data
theta4=zeros(1,5); %theta parameters
H4=theta4*X4; %forming the hypothesis
[JJ4 theta4]=Regression(H4,Train_y,X4,theta4);
%TEST----------------------------------------------------------
Xtest4=[ones(1,length(Test));(x1T.^(1/2))';(x2T.^(1/2))';(x3T.^2)';(x8T.^(1/2))'];
H_T4=theta4*Xtest4; 
J_test4=(1/(2*length(H_T4)))*sum((H_T4-Test_y').^2); 


