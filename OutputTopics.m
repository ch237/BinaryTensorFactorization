function [Topics]=OutputTopics(WP,WO,W_outputN)
%W_outputN: output terms for each topic
WP_sum=sum(WP);%total number of terms assigned to each topic
[T,Z]=size(WP);%T: # of terms;Z: # of topics
Topics=cell(Z,1);
for z=1:Z
    [WPsort TermIndex]=sort(WP(:,z),'descend');
    str='';
    for index=1:W_outputN       
        str=[str ' ' WO{TermIndex(index)}];
    end
    Topics{z}=str;
end



N=20;
[WPsort TermIndex]=sort(U{1}(:,165),'descend');
author_list=cell(N,1);
for index=1:N      
     author_list{index}=aname{TermIndex(index)};
end