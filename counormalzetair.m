function zetair = counormalzetair(U,id,lambda)
% R=size(U,2);
Nnon0=length(id{1,1});
% log_zetair=repmat(log(lambda+1),Nnon0,1)+2*log(U(id{1},:));
% zetair=repmat(lambda,Nnon0,1).*(U(id{1},:).^2);
zetair=repmat(lambda,Nnon0,1).*(U(id{1},:).*U(id{2},:));
% zetair=exp(log_zetair);
% zetair=max(zetair,1e-5);
end