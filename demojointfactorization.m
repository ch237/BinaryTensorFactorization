%%
% For cold start problem in which tensor data corresponding to some
% entities are completely missing, the goal is to predict those missing
% tensor values
%%
clear all;close all;
% load facebook_xiid
load duke_xiid

R=100;
numiters=50;
isnetwork=0;%use network or not
stentity=1000;% all entities with index larger than stentity are used as test data, this value should be smaller than the 1st dimentionality of tensor
fraction=5;% ratio between number of zeros and ones in testing data
for k=1:3 % adjust # of modes according to data
    N(k) = max(id{k}); 
end
% N=[63731,63731,1847];
isbatch=0;
co_isbatch=0;
batchsize=50000;
co_batchsize=50000;

[U lambda pr co_lambda co_pr eva time_trace]=...
    BTF_JointBinaryOnlineGibbsRandomInit(N,co_xi,co_id,xi,id,R,batchsize,co_batchsize,numiters,isbatch,co_isbatch,isnetwork,stentity,fraction);

%%
%for duke scholar data 
WO=textread('vocabnew.txt','%s');
W_outputN=100;
[Topics]=OutputTopics(U{1,2},WO,W_outputN);
[aid aname]=textread('aidanamenew.txt','%d%s','delimiter',':');
aidshow=1126;
[facval facid]=sort(U{1}(1126,:),'descend');
facshow=facid(1);
[facval facaid]=sort(U{1}(:,facshow),'descend');
showN=20;
for i=1:showN
    fprintf('%d %s\n',facaid(i),aname{facaid(i)});
end