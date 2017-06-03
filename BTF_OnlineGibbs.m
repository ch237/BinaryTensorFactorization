function [U lambda pr eva time_trace] = BTF_OnlineGibbs(N,xi,id,R,batchsize,numiters,isbatch,fraction,trainfraction)
    K=length(id);
%     for k=1:K 
%         N(k) = max(id{k}); 
%     end
    rng(0);
    % initial parameters of the dirichlet on U
    a=5e-1*ones(1,K);
    %a=ones(1,K);
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
    epsi=0.9;%1/R;    
    pr_a = c*epsi;
    pr_b = c*(1-epsi);
    pr=betarnd(pr_a,pr_b);
    
    % initial parameters of the gamma on lambda
    gr=0.1;
    lambda_a = gr*ones(1,R);
    lambda_b = (pr/(1-pr))*ones(1,R);
    lambda=gamrnd(gr,pr/(1-pr),1,R); 
    
    Nnon0=length(id{1,1});
    Train= floor(trainfraction*Nnon0);
    idall=randperm(Nnon0);
    idtest=cell(1,K);
    for k=1:K
        idtest{k} = id{k}(idall(Train+1:end)); 
    end  
    [idtest xi_test]=addzero(xi(idall(Train+1:end)),idtest,fraction);
%     save bi_test_face idtest xi_test
%     load bi_test_face
    perc=1;% fraction of total training data
    
    iterN=numiters;
    if isbatch
        Np = Train;
    else
        Np=batchsize;
%         Np=floor(Train*perc*0.1);
    end
    llikevec=zeros(iterN,1);
    t0 = 0;
    for iter=1:iterN
        tic;
        if isbatch
            gam_t = 1;
        else
            gam_t = (iter+t0)^(-0.5);
        end
        
        %if iter==1
        idselect = [];
%         n_tmp = randperm(Train);
        n_tmp = randperm(floor(Train*perc));
        idid = n_tmp(1:Np);
        idid=idall(idid);

        xiselect = xi(idid);
        for k=1:K
            idselect(:, k) = id{1, k}(idid);      
        end
        %end

        % compute the mean of xir's
        if iter==1
            zetair=unormalzetair(U,idselect,lambda);
            zetai=sum(zetair,2);
            xiselectlatent=zetai.*exp(zetai)./(exp(zetai)-1);% expectation of zero truncated poisson distribution
            %xiselectlatent=xiselect'.*truncated_Poisson_rnd(sum(zetair,2));
            xiselectlatent=truncated_Poisson_rnd_1(xiselect',sum(zetair,2));
            xiselectr=repmat(xiselectlatent,1,R).*zetair./repmat(sum(zetair,2),1,R);
            [xsumr,xr]=tensorsum(xiselectr,idselect,N);
            % update the sufficient statistics using the xir's
            % note: multiplying by Train/Np is a hack but seems to give
            % better result and faster convergence
            xr=xr*Train/Np;
            for k=1:K
                xsumr{k}= xsumr{k}*Train/Np;
            end
        else
            zetair=unormalzetair(U,idselect,lambda);
            xiselectlatent=truncated_Poisson_rnd_1(xiselect',sum(zetair,2));
            xiselectr=repmat(xiselectlatent,1,R).*zetair./repmat(sum(zetair,2),1,R);
            [xsumr_temp,xr_temp]=tensorsum(xiselectr,idselect,N);

            % update the sufficient statistics using the xir's
            % note: multiplying by Train/Np is a hack but seems to give
            % better result and faster convergence            
            xr = (1-gam_t)*xr_old + gam_t*xr_temp*Train/Np;
            for k=1:K
                xsumr{k}=(1-gam_t)*xsumr_old{k} + gam_t*xsumr_temp{k}*Train/Np;
            end
        end

        % update pr
        pr_a = c*epsi+xr;
        pr_b = c*(1-epsi)+gr;
        pr = pr_a./(pr_a+pr_b);     

        % update lambda
        lambda_a = gr+xr;
        lambda_b = pr;
        lambda = lambda_a.*lambda_b;
        
        % update U
        for r=1:R
            for k=1:K
                dir_a{k,r}(idselect(:, k)) = a(k)+xsumr{1,k}(idselect(:, k),r);
                U{1,k}(:,r) = dir_a{k,r}'/sum(dir_a{k,r}); 
                % TBD: U changed. Should we re-estimate the sufficient statistics?               
             end       
        end
        xr_old = xr;
        xsumr_old = xsumr;

        if iter==1 
            time_trace(iter) = toc;
        else
            time_trace(iter) = time_trace(iter-1) + toc;
        end 
 
        [llike mae rmse mse auc_test auc_pr_test]=evaluation_1(xi_test,idtest,U,lambda);
        llikevec(iter)=llike;rmsevec(iter)=rmse;maevec(iter)=mae;msevec(iter)=mse;auc(iter)=auc_test;auc_pr(iter)=auc_pr_test;
        fprintf('iter= %d;loglike= %f, mae=%f, mse=%f, rmse=%f, auc=%f, auc_pr=%f, time= %f\n', iter, llike, mae, mse, rmse, auc_test,auc_pr_test,time_trace(iter));
           
        subplot(3,1,1); plot(time_trace(1:iter),llikevec(1:iter));
        xlabel('Time (seconds)');
        ylabel('Heldout log-likelihood'); 
        subplot(3,1,2);plot(sort(lambda,'descend'));
        xlabel('Weights of rank-1 components-Tensor');
        subplot(3,1,3); plot(time_trace(1:iter),auc(1:iter));
        xlabel('Time (seconds)');
        ylabel('AUC'); 
        drawnow;
    end
    eva=[llikevec';rmsevec;maevec;msevec;auc;auc_pr];
end