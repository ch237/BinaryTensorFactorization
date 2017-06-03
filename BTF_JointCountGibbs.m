% function [U lambda pr co_lambda co_pr llikevec time_trace] = BTF_JointOnlineGibbs(co_xi,co_id,xi,id,xi_test,idtest,R,batchsize,numiters,isbatch)
function [U lambda pr co_lambda co_pr llikevec time_trace] = BTF_JointCountGibbs(co_xi,co_id,xi,id,R,numiters)
    K=length(id);
    for k=1:K 
        N(k) = max(id{k}); 
    end
    rng(0);
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
    c=1;
    epsi=1/R;    
    pr_a = c*epsi;
    pr_b = c*(1-epsi);
    pr=betarnd(pr_a,pr_b);
%     pr=pr_a/(pr_a+pr_b);
    
    co_pr=pr;
    
    % initial parameters of the gamma on lambda
    gr=0.1;
    lambda_a = gr*ones(1,R);
    lambda_b = (pr/(1-pr))*ones(1,R);
    lambda=gamrnd(gr,pr/(1-pr),1,R);
%     lambda=gr*pr/(1-pr)*ones(1,R);
    co_lambda=lambda;
    
    Nnon0=length(id{1,1});
%     Train= Nnon0;
    Train=floor(0.95*Nnon0);
    idall=randperm(Nnon0);
    idtest=cell(1,K);
    for k=1:K
        idtest{k} = id{k}(idall(Train+1:end)); 
    end  
    co_Nnon0=length(co_id{1});
    co_Train=floor(0.95*co_Nnon0);
    co_idall=randperm(co_Nnon0);
    co_idtest=cell(1,2);
    for k=1:2
        co_idtest{k} = co_id{k}(co_idall(co_Train+1:end)); 
    end  
    
    perc=1;% fraction of total training data
    
    iterN=numiters;
    llikevec=zeros(iterN,1);
    t0 = 0;
    tic;
    for iter=1:iterN
        if iter==1       
            idselect = [];
            idid=idall(1:Train);
            xiselect = xi(idid);
            for k=1:K
                idselect(:, k) = id{1, k}(idid);      
            end

            co_idselect =cell(1,2);
            co_idid=co_idall(1:co_Train);
            co_xiselect = co_xi(co_idid);
            for k=1:2
                co_idselect{k}= co_id{1, k}(co_idid);      
            end
        end
        
        
        zetair=computezetair_new(U,idselect,lambda);
        xiselectr=mnrnd(xiselect,zetair);         
        [xsumr,xr]=tensorsum(xiselectr,idselect,N);        
        
        co_zetair=counormalzetair(U{1},co_idselect,co_lambda);
        co_zetai=sum(co_zetair,2);
        co_xiselectlatent=truncated_Poisson_rnd_1(co_xiselect',co_zetai);
        co_xiselectr=mnrnd(co_xiselectlatent,co_zetair./repmat(co_zetai,1,R)); 
        [co_xsumr,co_xr]=cotensorsum(co_xiselectr,co_id,N(1));
        
        for r=1:R
            for k=1:K
                if k==0
                    dir_a{k,r}(idselect(:, k)) = a(k)+xsumr{1,k}(idselect(:, k),r);
                    dir_a{k,r}(co_id{1}(:,k))=dir_a{k,r}(co_id{1}(:,k))+co_xsumr{1}(co_id{1}(:,k),r);
                else
                    dir_a{k,r}(idselect(:, k)) = a(k)+xsumr{1,k}(idselect(:, k),r);
                end
                U{1,k}(:,r) = sampleDirMat(dir_a{k,r}',1)';
            end
        end
        
        pr=betarnd(c*epsi+xr,c*(1-epsi)+gr);
        lambda=gamrnd(gr+xr,pr);

        
        co_pr_a = c*epsi+co_xr;
        co_pr_b = c*(1-epsi)+gr;
        co_pr = betarnd(co_pr_a,co_pr_b);

        
        co_lambda_a = gr+co_xr;
        co_lambda_b = co_pr;
        co_lambda = gamrnd(co_lambda_a,co_lambda_b);

        if iter==1 
            time_trace(iter) = toc;
            tic;
        else
            time_trace(iter) = time_trace(iter-1) + toc;
            tic;
        end 
 
%         [llike mae rmse mse auc_test]=evaluation_1(xi_test',idtest,U,lambda);
        [llike mae rmse mse]=evaluation_pois(xi(idall(Train+1:end)),idtest,U,lambda);
%         llike=loglike(xi(idall(Train+1:end)),idtest,U,lambda);
        llikevec(iter)=llike;
%         rmse=rmseBayes(xi(idall(Train+1:end)),idtest,U,lambda);
        rmsevec(iter)=rmse;
        maevec(iter)=mae;
        msevec(iter)=mse;
       fprintf('iteration= %d;loglikelihood= %f, mae=%f, mse=%f, rmse=%f, time elapsed= %f\n', iter, llike, mae, mse, rmse, time_trace(iter));
%        fprintf('iteration= %d;loglikelihood= %f, test AUC=%f, MAE = %f, time elapsed= %f\n', iter, llike, auc_test, mae, time_trace(iter));
           
        %llike=loglike(xi(idall(Train+1:end)),idtest,U,lambda);
        %llikevec(iter)=llike;
        subplot(3,1,1); plot(time_trace(1:iter),llikevec(1:iter));
        xlabel('Time (seconds)');
        ylabel('Heldout log-likelihood'); 
        drawnow;
        subplot(3,1,2);plot(sort(lambda,'descend'));
        xlabel('Weights of rank-1 components-Tensor');
        subplot(3,1,3);plot(sort(co_lambda,'descend'));
        xlabel('Weights of rank-1 components-Matrix');
        %ylabel('Heldout log-likelihood');        
        drawnow;
        %fprintf('iteration= %d;loglikelihood= %f, time elapsed= %f\n', iter, llike, time_trace(iter)); 
    end
end