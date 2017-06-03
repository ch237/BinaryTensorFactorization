function [idmix ximix]=addzero(xi,id,fraction)
rng(0);
K=length(id);
minid=zeros(K,1);
maxid=zeros(K,1);
for k=1:K
    minid(k)=min(id{k});
    maxid(k)=max(id{k});
end
onedict = containers.Map;
for i=1:length(xi)
    str=num2str(id{1}(i));
    if mod(i,100000)==0
        fprintf('generate %d keys\n',i);
    end
    for k=2:K
        str=[str,',',num2str(id{k}(i))];
    end
    onedict(str)=1;
end
idzerolong=cell(1,K);
for k=1:K;
    idzerolong{k}=minid(k)-1+ceil((maxid(k)-minid(k))*(rand(floor(2*fraction*length(xi)),1)));
end
idzero=cell(1,K);
for k=1:K;
    idzero{k}=zeros(floor(fraction*length(xi)),1);
end
xizero=zeros(floor(fraction*length(xi)),1);
zerodict=containers.Map;
i=0;
for j=1:(floor(2*fraction*length(xi)))
    str=num2str(idzerolong{1}(j));
    for k=2:K
        str=[str,',',num2str(idzerolong{k}(j))];
    end
    if ~isKey(onedict,str) & ~isKey(zerodict,str)
       zerodict(str)=0; 
       i=i+1;
       for k=1:K
           idzero{k}(i)=idzerolong{k}(j);
       end
    end
    if i==floor(fraction*length(xi))
        break;
    end
    if mod(i,200000)==0
        fprintf('generate %d zeros\n',i);
    end
end

idmix=cell(1,K);
for k=1:K;
    idmix{k}=[id{k};idzero{k}];
end
ximix=[xi;xizero];


