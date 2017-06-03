clear all;
load face91splitbatch
auc_batch=eva(5,:);
time_batch=time_trace;
load face91splitonline
auc_online=eva(5,:);
time_online=time_trace;
figure(1),semilogx(time_batch,auc_batch,'r','LineWidth',1.5);
hold on;
semilogx(time_online,auc_online,'g','LineWidth',1.5);
title('Facebook Data','FontSize',24)
xlabel('Time in seconds','FontSize',24)
ylabel('AUC-ROC','FontSize',24)
legend('ZTP-CP (Batch MCMC)','ZTP-CP (OnlineMCMC)')
set(gca,'FontSize',18)

clear all;
load duke91splitbatch
auc_batch=eva(5,:);
time_batch=time_trace;
load duke91splitonline
auc_online=eva(5,:);
time_online=time_trace;
figure(2),semilogx(time_batch,auc_batch,'r','LineWidth',1.5);
hold on;
semilogx(time_online,auc_online,'g','LineWidth',1.5);
title('Scholar Data','FontSize',24)
xlabel('Time in seconds','FontSize',24)
ylabel('AUC-ROC','FontSize',24)
legend('ZTP-CP (Batch MCMC)','ZTP-CP (OnlineMCMC)')
set(gca,'FontSize',18)

% set(gca, 'ytick', 0.95:0.01:0.99)
% xlabel('Time in seconds','FontSize',24,'fontweight','b')
% ylabel('AUC-ROC','FontSize',24,'fontweight','b')
% set(gca,'FontSize',20,'fontweight','b')
