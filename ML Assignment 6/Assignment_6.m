clear
close all
clc
%%%%%%%%%%%%% reading data%%%%%%%%%%%%%%%%%%%%%%%%
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T=read(ds);

%%%%%%%%%%%%%%%% normalizing data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=T{:,4:21};
[m,n]=size(x);
Y=T{:,3}/max(T{:,3});
Data_Scaled=zeros(size(x));
for i=1:size(x,2)
    Data_Scaled(:,i)=x(:,i)/max(x(:,i));
end

% %%%%%%%%%%%%%% calculating correlation and covariance matricies%%%%%%%%%%
Corr_x=corr(Data_Scaled);
x_cov=cov(Data_Scaled);

% %%%%%%%%%% pricipal components analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U S V]=svd(x_cov); 
EigenValues=diag(S)'; % V is the eigen vectors and S is the eigen values
k=1;
while(true)
    alpha=1-(sum(EigenValues(1:k))/sum(EigenValues(1:18)));
    if(alpha <= 0.001)
        break;
    end
    k=k+1;
end
Reduced_Data=U(:,1:k)'* Data_Scaled';
App_Data=Reduced_Data'*V(1:k,:); %reduced data multiplied by the eigen vectors
Error=(1/17999).* sum((App_Data(:,1:k)'-Reduced_Data).^2); % error between approx data and reduced data.

% %%%%%%% linear regresion on reduced data %%%%%%%%%%%%%%%%%%%%%%%%%
X=[ones(1,length(Reduced_Data));Reduced_Data];
theta1=zeros(1,k+1); %theta parameters
H1=theta1*X; %forming the hypothesis
[JJ1 theta1]=Linear_Regression(H1,Y,X,theta1);

% %%%%%%%%% K-Means Clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%

% %cluster centroid initialization
MIN=[];
%clustering on the reduced data.
for K=1:10
    J=[];
    C_Optimal=[];
    miu_Optimal=[];
    miu_ci_Optimal=[];
    for i=1:100
        [C,miu,miu_ci]=K_Means(Reduced_Data,K);
        J=[J sum(sum((Reduced_Data-miu_ci).^2))/length(Reduced_Data)];
    end
    MIN(K)=min(J);
end
K=1:10;
figure();
plot(K,smooth(MIN));
title('K-Means Clustering on the Reduced Data');
xlabel('Number of Clusters');
ylabel('Distortion Function');
% %--------------------------------------------------------------------------
MIN2=[];
Real_Data=Data_Scaled';
%clustering on the real house data
for K2=1:10
J2=[];
C2_Optimal=[];
miu2_Optimal=[];
miu2_ci_Optimal=[];
for i=1:100
    [C2,miu2,miu2_ci]=K_Means(Real_Data,K2);
    J2=[J2 sum(sum((Real_Data-miu2_ci)).^2)/length(Real_Data)];
end
MIN2(K2)=min(J2);
end
K2=1:10;
figure();
plot(K2,smooth(MIN2));
title('K-means Clustering on The Real Data');
xlabel('Number of Clusters');
ylabel('Distortion Function');

%%%%%%%%%%%%%%%%%%%%%% Anomally Detection %%%%%%%%%%%%%%%%%
Mean=mean(x);
Sigmaormcdf(x(j,i),Mean,Sigma);
 
end
epsilon=std(x);
j=9;
for i=1:n
    
P=n=0.001;
AnomallyDetection=(prod(P)<epsilon||prod(P)>(1-epsilon));