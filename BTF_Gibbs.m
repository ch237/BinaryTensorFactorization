%function [U lambda pr llikevec mae time_trace rankvec] = PTF_Gibbs(xi,id,R)
function [U lambda pr llikevec mae time_trace rankvec] = BTF_Gibbs(xi,id,xi_test,idtest,R,numiters,burnin)
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

    iterN=numiters;
    %burnin=300;
    step=4;
    sampleN=(iterN-burnin)/step;
    % samPsi=zeros(size(Psi));

    xitrain=xi;
    %idtrainvec=idtrain;
    idtrain=cell(1,K);
    for k=1:K
        idtrain{k} = id{k};
        idtest{k} = idtest{k};
    end
    llikevec=zeros(iterN,1);

    zetair=unormalzetair(U,idtrain,lambda);
%     xitrainlatent=truncated_Poisson_rnd(sum(zetair,2));
    xitrainlatent=truncated_Poisson_rnd_1(xitrain,sum(zetair,2));
    xir=mnrnd(xitrainlatent,zetair./repmat(sum(zetair,2),1,R));

    llikevec=zeros(iterN,1);
    rmsevec=zeros(iterN,1);
    maevec=zeros(iterN,1);
    msevec=zeros(iterN,1);
    r=zeros(iterN,1);

    tic
    for iter=1:iterN
    %     iter
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
        %xitrainlatent=truncated_Poisson_rnd(sum(zetair,2));
        xitrainlatent=truncated_Poisson_rnd_1(xitrain,sum(zetair,2));
        xir=mnrnd(xitrainlatent,zetair./repmat(sum(zetair,2),1,R));
    %     xi=xigenerate(X,id);
%         xir=mnrnd(xitrain,zetair);
        if iter>burnin & mod(iter-burnin,step)==0
            if iter==(burnin+step)
                lambdasam=lambda/sampleN;
                Usam=U;
                for k=1:K
                    Usam{1,k}=Usam{1,k}/sampleN;                
                end
            else
                lambdasam=lambdasam+lambda/sampleN;
                for k=1:K
                    Usam{1,k}=Usam{1,k}+U{1,k}/sampleN;                
                end
            end
        end

            if iter==1 
                time_trace(iter) = toc;
                tic;
            else
                time_trace(iter) = time_trace(iter-1) + toc;
                tic;
            end 

        [llike mae rmse mse auc_test]=evaluation_1(xi_test',idtest,U,lambda);
        llikevec(iter)=llike;
        rmsevec(iter)=rmse;
        maevec(iter)=mae;
        msevec(iter)=mse;
       %fprintf('iteration= %d;loglikelihood= %f, mae=%f, mse=%f, rmse=%f, time elapsed= %f\n', iter, llike, mae, mse, rmse, time_trace(iter));
       fprintf('iteration= %d;loglikelihood= %f, test AUC=%f, MAE = %f, time elapsed= %f\n', iter, llike, auc_test, mae, time_trace(iter));
           
        %llike=loglike(xi(idall(Train+1:end)),idtest,U,lambda);
        %llikevec(iter)=llike;
        subplot(2,1,1); plot(time_trace(1:iter),llikevec(1:iter));
        xlabel('Time (seconds)');
        ylabel('Heldout log-likelihood'); 
        drawnow;
        subplot(2,1,2);plot(sort(lambda,'descend'));
        xlabel('Weights of rank-1 components');
        %ylabel('Heldout log-likelihood');        
        drawnow;
        %fprintf('iteration= %d;loglikelihood= %f, time elapsed= %f\n', iter, llike, time_trace(iter)); 
    end
end