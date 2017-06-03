function [llike mae rmse mse auc_test auc_test2]=evaluation(xi,id_temp,U,lambda)
K=length(U);
if iscell(id_temp)
    id=id_temp;
else    
id=cell(1,K);
    for k=1:K
        id{1,k}=id_temp(:,k);
    end
end

R=size(U{1,1},2);
Np=length(id{1,1});
zetair=ones(Np,R).*repmat(lambda,Np,1);
for k =1:K
    zetair=zetair.*U{1,k}(id{1,k},:);
end
zetai=sum(zetair,2);
% llike=sum(log(poisspdf(xi,zetai)+eps));
% llike=sum(-zetai+xi.*log(zetai)-log(factorial(xi)));
% llike=sum(-zetai+xi.*log(zetai)-gammaln(xi+1));
%llike=sum(log(1-exp(-zetai)));%sum(-zetai+xi.*log(zetai)-gammaln(xi+1));
%llike = sum(xi.*log(1-exp(-zetai)) + (1-xi).*log(exp(-zetai)));

llike = sum(xi.*log(1-exp(-zetai)) - (1-xi).*zetai);
rmse=norm(xi-(1-exp(-zetai)),'fro')/(sqrt(size(xi,1) * size(xi,2)));
mae=sum(abs(xi-(1-exp(-zetai))))/Np;
mse=rmse^2;
auc_test = compute_AUC(xi,1-exp(-zetai),ones(1,length(xi)));

%[prec, tpr, fpr, thresh] = prec_rec(1-exp(-zetai), xi,  'numThresh',3000);
[prec, tpr, fpr, thresh] = prec_rec(1-exp(-zetai), xi);
auc_test2 = trapz([0;tpr],[1;prec]);