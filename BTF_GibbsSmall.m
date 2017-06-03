function [U lambda pr eva time_trace] = BTF_GibbsSmall(xi,id,xi_tensor,R,numiters,burnin,trainfrac)
    K=length(id);
    for k=1:K 
        N(k) = max(id{k}); 
    end
    Nnon0=length(id{1,1});
    %Train=Nnon0;

    rng(0);
    a=5e-1*ones(1,K);
    U=cell(1,K);
    for k=1:K
        U{1,k} = sampleDirMat(a(k)*ones(1,N(k)),R);
        U{1,k} = U{1,k}';
    end

    c=1;
    epsi=1/R;
    pr=betarnd(c*epsi,c*(1-epsi));
    gr=0.1;
    % lambda=1/R.*ones(1,R);
    lambda=gamrnd(gr,pr/(1-pr),1,R);
    
    Nnon0=length(id{1,1});
    Train=floor(trainfrac*Nnon0);
%     idall=1:Nnon0;%
    idall=randperm(Nnon0);
    idtest=cell(1,K);
    idtrain=cell(1,K);
    for k=1:K
        idtest{k} = id{k}(idall(Train+1:end));
        idtrain{k} = id{k}(idall(1:Train));
    end
    xitrain=ones(Train,1);
    zeroN=ceil((Nnon0-Train)*numel(xi_tensor)/length(xi));
    zeroID=cell(1,K);
    if K==3      
        [zeroID{1},zeroID{2},zeroID{3}] = ind2sub(size(xi_tensor),find(xi_tensor== 0));
    end   
    if K==2
        [zeroID{1},zeroID{2}] = ind2sub(size(xi_tensor),find(xi_tensor== 0));
    end
    zeroIDid=randperm(length(zeroID{1}));
    zeroIDid=zeroIDid(1:zeroN);
    for k=1:K
        idtest{k}=[idtest{k};zeroID{k}(zeroIDid)];
    end
    xi_test=[ones(Nnon0-Train,1);zeros(zeroN,1)];
    zetaitest=zeros(size(xi_test));
    

    iterN=numiters;
    %burnin=300;
    step=5;
    sampleN=(iterN-burnin)/step;

    llikevec=zeros(iterN,1);

    zetair=unormalzetair(U,idtrain,lambda);
%     xitrainlatent=truncated_Poisson_rnd(sum(zetair,2));
    xitrainlatent=truncated_Poisson_rnd_1(xitrain',sum(zetair,2));
    xir=mnrnd(xitrainlatent,zetair./repmat(sum(zetair,2),1,R));

    llikevec=zeros(iterN,1);
    rmsevec=zeros(iterN,1);
    maevec=zeros(iterN,1);
    msevec=zeros(iterN,1);
    r=zeros(iterN,1);

%     tic
    for iter=1:iterN
    %     iter
        tic
        [xsumr,xr]=tensorsum(xir,idtrain,N);
        pr=betarnd(c*epsi+xr,c*(1-epsi)+gr);
        for r=1:R
            for k=1:K
                U{1,k}(:,r) = sampleDirMat(a(k)+xsumr{1,k}(:,r)',1)';
            end
        end
        lambda=gamrnd(gr+xr,pr);
        %zetair=computezetair_new(U,idtrain,lambda);
        zetair=unormalzetair(U,idtrain,lambda);
        zetai=sum(zetair,2);
        %xitrainlatent=truncated_Poisson_rnd(sum(zetair,2));
        xitrainlatent=truncated_Poisson_rnd_1(xitrain',zetai);
        xir=mnrnd(xitrainlatent,zetair./repmat(zetai,1,R));

            if iter==1 
                time_trace(iter) = toc;
%                 tic;
            else
                time_trace(iter) = time_trace(iter-1) + toc;
%                 tic;
            end 

        [llike mae rmse mse auc_test auc_pr_test zetaitest_tmp]=evaluation_zetai(xi_test,idtest,U,lambda);
        llikevec(iter)=llike;rmsevec(iter)=rmse;maevec(iter)=mae;msevec(iter)=mse;auc(iter)=auc_test;auc_pr(iter)=auc_pr_test;
        fprintf('iter= %d;loglike= %f, mae=%f, mse=%f, rmse=%f, auc=%f, auc_pr=%f, time= %f\n', iter, llike, mae, mse, rmse, auc_test,auc_pr_test,time_trace(iter));
       if iter>burnin & mod(iter-burnin,step)==0
            if iter==(burnin+step)
                zetaitest=zetaitest_tmp/sampleN;
            else
                zetaitest=zetaitest+zetaitest_tmp/sampleN;
            end
        end
           
        subplot(3,1,1); plot(time_trace(1:iter),llikevec(1:iter));
        xlabel('Time (seconds)');
        ylabel('Heldout log-likelihood'); 
        subplot(3,1,2);plot(sort(lambda,'descend'));
        xlabel('Weights of rank-1 components');
        subplot(3,1,3); plot(time_trace(1:iter),auc(1:iter));
        xlabel('Time (seconds)');
        ylabel('AUC');   
        drawnow;
    end
    eva=[llikevec';rmsevec';maevec';msevec';auc];
    auc_sam = compute_AUC(xi_test,zetaitest,ones(1,length(xi_test)));
    fprintf('Final test AUC=%f\n', auc_sam);
end