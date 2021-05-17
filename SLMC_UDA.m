function [Acc,Yt_pred] = SLMC_UDA(Xs,Ys,Xt,Yt,options)

% Reference:


%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
    fprintf('SLMC starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
 


   
    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        
        %% frist
        W = lapgraph(X',manifold);
        W(1:n,1:n)=0;
        %% second
        W(1:n,n+1:end)=0;
        W(n+1:end,1:n)=0;

        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
        
    else
        L = 0;
    end
    
  
  
    % Construct kernel
    K = kernel_cal('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    %E = 
    V = diag(sparse([ones(n,1);zeros(m,1)]));

        
       
   
        % Compute coefficients vector Beta
    Beta = ((V + options.lambda * (L)) * K + options.eta * speye(n + m,n + m)) \ (V * YY);
    F = K * Beta;
	[~,Cls] = max(F,[],2);
	%% Compute accuracy
	Acc = numel(find(Cls(n+1:end)==Yt)) / m;
	Cls = Cls(n+1:end);
    Yt_pred = Cls;
    fprintf('SLMC_UDA ends!\n');
end

function V = updateV(n,m,Yt,F,ratio)
    Y = ones(m,1);
	ind = (sub2ind([m,size(F,2)],1:m,Yt'));
	tmp = (Y - F(ind)').^2;
	order = sort (tmp);
	M = tmp<order(ceil(ratio*m));
	V = diag(sparse([ones(n,1);M]));
 
end

function K = kernel_cal(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end