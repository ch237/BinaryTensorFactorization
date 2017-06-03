clear all;close all;
% load facebook_xiid
load duke_xiid

R=100;
numiters=60;
fraction=5;% ratio between number of zeros and ones in testing data
for k=1:3 
    N(k) = max(id{k}); 
end

trainfraction=0.9;%90 percent as training data
isbatch=0;% 1: batch gibbs; 0: online gibbs
batchsize=floor(length(id{1})*trainfraction/10);%9/1 training/test split, batch size is 10 percent of training data

[U lambda pr eva time_trace] = BTF_OnlineGibbs(N,xi,id,R,batchsize,numiters,isbatch,fraction,trainfraction);
% WO=textread('vocabnew.txt','%s');
% W_outputN=100;
% [Topics]=OutputTopics(U{1,2},WO,W_outputN);
% [aid aname]=textread('aidanamenew.txt','%d%s','delimiter',':');
% aidshow=1126;
% [facval facid]=sort(U{1}(1126,:),'descend');
% facshow=facid(1);
% [facval facaid]=sort(U{1}(:,facshow),'descend');
% showN=20;
% for i=1:showN
%     fprintf('%d %s\n',facaid(i),aname{facaid(i)});
% end