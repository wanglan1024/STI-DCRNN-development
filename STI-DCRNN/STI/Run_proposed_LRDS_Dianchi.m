% ========================================================================
% Spatio-Temporal Signal Recovery Based on Low Rank and Differential Smoothness
%
% Copyright(c) 2018 Kai Qiu, Xianghui Mao and Yuantao Gu
% All Rights Reserved.
% ----------------------------------------------------------------------
% Batch Reconstruction of Time-Varying Graph Signal (BR-TVGS), objective function:
% f(x)= 0.5* ||J*x-y||^2 + 0.5*alpha* (Tx)'*L*(Tx) + mu* ||x||_* ,
%  
% Version 1.0
% Written by Kai Qiu (q1987k@163.com)
% Modified by Xianghui Mao (maoxh92@sina.com) 
%----------------------------------------------------------------------

clear all;close all;clc;
addpath('../../solvers');
addpath('../../utilities');
load Chla.mat
load Position.mat

Data = Chla().';

[N,T] = size(Data);
Data_original = Data; 
Data(isnan(Data)) = 0;
SampleMatrix0 = Data ~= 0; 

Dist = zeros(N);
for i = 1:N
     for j = i+1:N
        Dist(i,j)=distance(Position(i,1),Position(i,2),Position(j,1),Position(j,2));
        %Dist(i,j) = Dist(i,j) * (0.5+ abs(Position(i,1) - Position(j,1)) / (abs(Position(i,1) - Position(j,1)) + abs(Position(i,2) - Position(j,2))));
        Dist(j,i)=Dist(i,j);
    end
end

A=zeros(N);
W=zeros(N);
k=2; 
for i=1:N
    [sortd,ind]=sort(Dist(i,:),'ascend');  
    A(i,ind(2:k+1))=1;
    A(ind(2:k+1),i)=1;
    W(i,ind(2:k+1))=1./(Dist(i,ind(2:k+1))).^2;
    W(ind(2:k+1),i)=W(i,ind(2:k+1));
end
WW=W;
W=W/max(max(W));
figure;gplot(A,Position,'*-');
% 
D=diag(sum(W));
L=D-W;

[V,Lambda]=eig(L); 
Lambda(1,1)=0;
lambda=diag(Lambda);
save paramAWD A W D L
load paramAWD

test_time =2;
srate = [0.7];

errorMAE_STI = zeros(test_time,length(srate));
errorRMSE_STI= zeros(test_time, length(srate));
mae_a = zeros(test_time, length(srate));
rmse_a = zeros(test_time, length(srate));
mae_m = zeros(test_time, length(srate));
rmse_m = zeros(test_time, length(srate));
data_recon = zeros(test_time, length(srate));
R2 = zeros(test_time, length(srate));

for k1 = 1:length(srate)
    for k2 = 1:test_time
        disp(['sampling rate ', sprintf('%.2f', srate(k1)), ', running test ', sprintf('%d', k2)]);
        SampleNum = floor(N*srate(k1));
        SampleMatrix = zeros(N,T);
    for i = 1:T
        IndexMeasure = find(SampleMatrix0(:,i));
        availableCount = length(IndexMeasure);
        if availableCount > 0
            actualSampleNum = min(SampleNum, availableCount);
            perm = randperm(availableCount);
            IndexSelect = IndexMeasure(perm(1:actualSampleNum));
            SampleMatrix(IndexSelect,i) = 1;
        end
    end
    SampleMatrix_res = SampleMatrix0 - SampleMatrix;
     
        %% Reconstruction
        param.J = SampleMatrix(:,1:T); 
        param.y = param.J .* (Data+sqrt(0)*randn(size(Data)));
        param.L = L;
        param.T = TV_Temp();  
        param.niter = 2000;
        param.display = 1;
               
        Mu_set = [ 0.01 0.02 0.05 0.1 0.5 1 2 5 10];
        alpha = [1e-2,1e-1,1,10 100 300];
        errorSeekmae = zeros(length(alpha),length(Mu_set));
        errorSeekrmse= zeros(length(alpha),length(Mu_set));
        
        for j_la = 1 : length(alpha)
            param.alpha = alpha(j_la); 
            param.belta = 0; 
            x_recon = param.y;
            param.z = 0 * x_recon;
            param.w = 0 * x_recon;
        
            x_recon = CGsolver_LRDS(x_recon, param);
            z_init = x_recon;
            for i_mu = 1 : length(Mu_set)
                tic
                param.alpha = alpha(j_la);
                param.mu = Mu_set(i_mu);
                param.belta = 1;
                param.outer = 50;
                param.z = z_init;
                param.w = 0 * x_recon;
                for k=1:param.outer
                    x_recon = CGsolver_LRDS(x_recon, param);
                    [uu,ss,vv] = svd(x_recon + param.w/param.belta);
                    ss = sign(ss) .* max(abs(ss)-param.mu/param.belta, 0);
                    param.z = uu * ss * vv';
                    param.w = param.w + param.belta * (x_recon - param.z);
                end
                toc
                diff = abs(Data(SampleMatrix_res>0) - x_recon(SampleMatrix_res>0));
                errorSeekrmse(j_la,i_mu)=sqrt(mean(abs(Data(SampleMatrix_res>0) - x_recon(SampleMatrix_res>0)).^2));
                errorSeekmae(j_la,i_mu) = mean(diff);
             end
        end
         [minrmse,I]=min(errorSeekrmse(:));
         [rmse_alpha,rmse_mu] = ind2sub(size(errorSeekrmse),I);
         errorRMSE_STI(k2,k1) = minrmse;
         [minmae,Ii]=min(errorSeekmae(:));
         [mae_alpha,mae_mu] = ind2sub(size(errorSeekmae),Ii);
         errorMAE_STI(k2,k1) = minmae;
         rmse_a(k2,k1) = rmse_alpha;
         rmse_m(k2,k1) = rmse_mu;
         mae_a(k2,k1) = mae_alpha;
         mae_m(k2,k1) = mae_mu;
         
    %for k2 = 2:test_time
%          disp(['K3:sampling rate ', sprintf('%.2f', srate(k1)), ', running test ', sprintf('%d', k2)]);
%         
%         SampleNum3 = floor(N*srate(k1));
%         SampleMatrix3 = zeros(N,T);
%         for i = 1:T
%             IndexMeasure = find(SampleMatrix0(:,i));
%             IndexSelect = IndexMeasure(randperm(length(IndexMeasure), SampleNum3));
%             SampleMatrix3(IndexSelect,i) = 1;
%         end
%         SampleMatrix_res3 = SampleMatrix0 - SampleMatrix3;

        %% Reconstruction
        tic
        param.J = SampleMatrix(:,1:T);
        param.y = param.J .* (Data+sqrt(0)*randn(size(Data)));
        param.L = L;
        param.T = TV_Temp();
        param.niter = 2000;
        param.display = 1;               
        param.alpha = alpha(rmse_alpha);
        param.belta = 0;
        x_recon = param.y;
        param.z = 0 * x_recon;
        param.w = 0 * x_recon;
        
        x_recon = CGsolver_LRDS(x_recon, param);
        

        param.alpha = alpha(rmse_alpha);
        param.mu = Mu_set(rmse_mu); 
        param.belta = 1;
        param.outer = 30;
        param.z = x_recon;
        param.w = 0 * x_recon;

        
        for k=1:param.outer  
            x_recon = CGsolver_LRDS(x_recon, param);
            [uu,ss,vv] = svd(x_recon + param.w/param.belta);
            ss = sign(ss) .* max(abs(ss)-param.mu/param.belta, 0);
            param.z = uu * ss * vv';
            param.w = param.w + param.belta * (x_recon - param.z);
        end
        toc
        x_recon01= x_recon
        diff = abs(Data(SampleMatrix_res>0) - x_recon(SampleMatrix_res>0));
        errorMAE_STI(k2,k1) = mean(diff); 
        errorRMSE_STI(k2,k1) = sqrt(mean(abs(Data(SampleMatrix_res>0) - x_recon(SampleMatrix_res>0)).^2));
        R=power(corrcoef(Data(SampleMatrix_res>0),x_recon(SampleMatrix_res>0)),2);
        R2(k2,k1) = R(1,2);
    end
    errorMAE_STI_v = errorMAE_STI;
    errorRMSE_STI_v = errorRMSE_STI;
    save errorMAE_RMSE errorMAE_STI_v errorRMSE_STI_v
    save Data_recon01 x_recon01
end


