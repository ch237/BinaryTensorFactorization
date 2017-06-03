%%
% for small datasets, the difference with the codes for larger dataset is
% that for small datasets, when evaluating performance, no need to generate
% zeros in testing data.
clear all;
%%
% load kinship_idxi
% load umls_idxi
% load nation_idxi
load movielens_idxi
%%
R=30;%rank
trainfrac=0.9;%90 percent as training data
batchsize=floor(length(xi)*trainfrac/10);%10 percent of training data as a batch
numiters=300;
isbatch=0;
burnin=floor(numiters*2/3);

%% choose online or batch gibbs for small data 
% [U lambda pr eva time_trace] = BTF_OnlineGibbsSmall(xi,id,xi_tensor,R,batchsize,numiters,isbatch,trainfrac); %online gibbs
[U lambda pr eva time_trace] = BTF_GibbsSmall(xi,id,xi_tensor,R,numiters,burnin,trainfrac);% batch gibbs