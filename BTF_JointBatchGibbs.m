% function [U lambda pr co_lambda co_pr llikevec time_trace] = BTF_JointOnlineGibbs(co_xi,co_id,xi,id,xi_test,idtest,R,batchsize,numiters,isbatch)
function [U lambda pr co_lambda co_pr eva time_trace] = BTF_JointBatchGibbs(N,co_xi,co_id,xi,id,R,numiters,isnetwork,stentity,fraction)
    K=length(id);
%     for k=1:K 
%         N(k) = max(id{k}); 
%     end
    rng(0);
    xi=ones(size(xi));
    % initial parameters of the dirichlet on U
    a=5e-1*ones(1,K);
    for k=1:K
        for r=1:R
            dir_a{k,r} = a(k)*ones(N(k),1);
        end
    end    
    U=cell(1,K);
    for k=1:K
        U{1,k} = sampleDirMat(a(k)*ones(1,N(k)),R);
        U{1,k} = U{1,k}';
    end
    
    % initial parameters of the beta on pr
    c=1;epsi=1/R;pr_a = c*epsi;pr_b = c*(1-epsi);
    pr=betarnd(pr_a,pr_b);co_pr=pr;
%     pr=pr_a/(pr_a+pr_b);co_pr=pr;
    
    % initial parameters of the gamma on lambda
    gr=0.1;lambda_a = gr*ones(1,R);lambda_b = (pr/(1-pr))*ones(1,R);
    lambda=gamrnd(gr,pr/(1-pr),1,R);co_lambda=lambda;
%     lambda=gr*pr/(1-pr)*ones(1,R);co_lambda=lambda;
    
    Nnon0=length(id{1,1});
%     Train= Nnon0;
    startid=find(id{1}==stentity);% index for the 1000th author
    Train=startid(1)-1;%floor(0.1*Nnon0);
    idall=1:Nnon0;%idall=randperm(Nnon0);
    idtest=cell(1,K);
    for k=1:K
        idtest{k} = id{k}(idall(Train+1:end)); 
    end
    
%     [idtest xitest]=addzero(xi(idall(Train+1:end)),idtest,fraction);
%     save bitest idtest xitest
    load bitest
    
    co_Nnon0=length(co_id{1});
    co_Train=floor(1*co_Nnon0); 
    perc=1;% fraction of total training data    
    iterN=numiters;
    Np = Train;   
    co_Np=co_Train;
    
    t0 = 0;
    tic;
    for iter=1:iterN
        
        idselect = [];
        idid=1:floor(Train*perc);
        idid=idall(idid);
        xiselect = xi(idid);
        for k=1:K
            idselect(:, k) = id{1, k}(idid);      
        end

        zetair=unormalzetair(U,idselect,lambda);          
        zetai=sum(zetair,2);
%             xiselectlatent=zetai.*exp(zetai)./(exp(zetai)-1);% expectation of zero truncated poisson distribution
        xiselectlatent=truncated_Poisson_rnd_1(xiselect',zetai);            
%             xiselectr=repmat(xiselectlatent,1,R).*zetair./repmat(sum(zetair,2),1,R);
        xiselectr=mnrnd(xiselectlatent,zetair./repmat(zetai,1,R));
        [xsumr,xr]=tensorsum(xiselectr,idselect,N);
         
        co_zetair=counormalzetair(U{1},co_id,co_lambda);
        co_zetai=sum(co_zetair,2);           
        co_xiselectlatent=truncated_Poisson_rnd_1(co_xi',co_zetai);
%         co_xiselectlatent=co_zetai.*exp(co_zetai)./(exp(co_zetai)-1);
%         co_xiselectr=repmat(co_xiselectlatent,1,R).*co_zetair./repmat(co_zetai,1,R);
        co_xiselectr=mnrnd(co_xiselectlatent,co_zetair./repmat(co_zetai,1,R));
        [co_xsumr,co_xr]=cotensorsum(co_xiselectr,co_id,N(1));
        % update pr
        pr_a = c*epsi+xr;pr_b = c*(1-epsi)+gr;pr=betarnd(pr_a,pr_b);%pr = pr_a./(pr_a+pr_b);       
        co_pr_a = c*epsi+co_xr;co_pr_b = c*(1-epsi)+gr;co_pr=betarnd(co_pr_a,co_pr_b);%co_pr = co_pr_a./(co_pr_a+co_pr_b);

        % update lambda
        lambda_a = gr+xr;lambda_b = pr;lambda=gamrnd(lambda_a,lambda_b);%lambda = lambda_a.*lambda_b;      
        co_lambda_a = gr+co_xr;co_lambda_b = co_pr;co_lambda=gamrnd(co_lambda_a,co_lambda_b);%co_lambda = co_lambda_a.*co_lambda_b;
        
        % update U
        for r=1:R
            for k=1:K
                dir_a{k,r}(idselect(:, k)) = a(k)+xsumr{1,k}(idselect(:, k),r);
                if isnetwork && k==1
                    dir_a{k,r}(co_id{1}(:,k))=dir_a{k,r}(co_id{1}(:,k))+co_xsumr{1}(co_id{1}(:,k),r);
                end                    
%                 U{1,k}(:,r) = dir_a{k,r}'/sum(dir_a{k,r});
                U{1,k}(:,r) = sampleDirMat(dir_a{k,r}',1)';
             end       
        end

        if iter==1 
            time_trace(iter) = toc;
            tic;
        else
            time_trace(iter) = time_trace(iter-1) + toc;
            tic;
        end 

%         [llike mae rmse mse]=evaluation(xi(idall(Train+1:end)),idtest,U,lambda);
%         [llike mae rmse mse]=evaluation(xitest,idtest,U,lambda);
        [llike mae rmse mse auc_test auc_pr_test]=evaluation_1(xitest,idtest,U,lambda);
        llikevec(iter)=llike;rmsevec(iter)=rmse;maevec(iter)=mae;msevec(iter)=mse;auc(iter)=auc_test;auc_pr(iter)=auc_pr_test;
        fprintf('iter= %d;loglike= %f, mae=%f, mse=%f, rmse=%f, auc=%f, auc_pr=%f, time= %f\n', iter, llike, mae, mse, rmse, auc_test,auc_pr_test,time_trace(iter));  
        
        subplot(4,1,1); plot(time_trace(1:iter),llikevec(1:iter));
        xlabel('Time (seconds)');
        ylabel('Heldout log-likelihood'); 
        subplot(4,1,2);plot(sort(lambda,'descend'));
        xlabel('Weights of rank-1 components-Tensor');
        subplot(4,1,3);plot(sort(co_lambda,'descend'));
        xlabel('Weights of rank-1 components-Matrix'); 
        subplot(4,1,4); plot(time_trace(1:iter),auc(1:iter));
        xlabel('Time (seconds)');
        ylabel('AUC'); 
        drawnow;
    end
    eva=[llikevec;rmsevec;maevec;msevec;auc;auc_pr];
end