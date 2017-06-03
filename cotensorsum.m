function [xsumr,xr]=cotensorsum(xir,id,N)
%N: # of enties
[Nnon0,R]=size(xir);
xsumr=cell(1,1);
xsumr{1}=zeros(N,R);

for i=1:Nnon0
    idtemp=id{1}(i);
    xsumr{1}(idtemp,:)=xsumr{1}(idtemp,:)+xir(i,:);
    idtemp=id{2}(i);
    xsumr{1}(idtemp,:)=xsumr{1}(idtemp,:)+xir(i,:);
end

xr=sum(xir,1);

