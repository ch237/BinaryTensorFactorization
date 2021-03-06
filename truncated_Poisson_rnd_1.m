function x = truncated_Poisson_rnd_1(xi,lambda)
% draw random samples from a truncated Poisson distribution
% P(x=k) = poisspdf(k,lambda)/(1-exp(-lambda))
% Coded by Mingyuan Zhou, 042014
%%Demo:
%lambda=0.3;aa=hist(truncated_Poisson_rnd(lambda*ones(1,10000)),1:20);
%plot(1:20,aa/sum(aa),'r',1:20,poisspdf(1:20,lambda)./(1-exp(-lambda)));

log1 = lambda>1 & xi'~=0;
log2 = lambda<=1 & xi'~=0;
lambda1=lambda(log1);
lambda2=lambda(log2);
x=zeros(length(lambda),1);
x1 = zeros(length(lambda1),1);
x2 = zeros(length(lambda2),1);

while 1
    dex=find(x1==0);
    if isempty(dex)
        break
    else
        lambdadex=lambda1(dex);
        temp = poissrnd(lambdadex);
        idex = temp>0;
        x1(dex(idex))=temp(idex);
    end
end
x(log1)=x1;

while 1
    dex=find(x2==0);
    if isempty(dex)
        break
    else
        lambdadex=lambda2(dex);
        temp = 1+poissrnd(lambdadex);
        idex = rand(size(temp))<1./(temp);
        x2(dex(idex))=temp(idex);
    end
end
x(log2)=x2;

