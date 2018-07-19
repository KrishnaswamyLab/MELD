%%
addpath(genpath('~/Documents/GitHub/MAGIC2/'));
addpath(genpath('~/Documents/GitHub/Blitz/'));

%% init
N_vec = [1000 2500 5000 10000 25000 50000];
t_vec = [6 6 12 24 48 48];
n_dim = 100;
n_branch = 50;
sigma = 5;
npca = 100;
ndim = 2;
time_vec = nan(size(N_vec));

%% run
for J=1:length(N_vec)
    J
    tic;
    
    n_samp = N_vec(J);
    
    % generate random fractal tree via DLA
    rng(17);
    n_steps = round(n_samp/n_branch);
    n_samp = n_branch * n_steps;
    M = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
    for I=1:n_branch-1
        ind = randsample(size(M,1), 1);
        M2 = cumsum(-1 + 2*(rand(n_steps,n_dim)),1);
        M = [M; repmat(M(ind,:),n_steps,1) + M2];
    end
    C = repmat(1:n_branch,n_steps,1);
    C = C(:);
    M = M + normrnd(0,sigma,size(M,1),size(M,2));
    
    % PCA
    pc = svdpca(M, npca, 'random');
    
    % operator
    P = compute_operator2(pc, 'k', 25);
    
    % fast power
    pc_t = pc;
    for I=1:t_vec(J)
        pc_t = P * pc_t;
    end
    %P_t = pc_t * pinv(pc);
    
    % potential
%     X = P_t;
%     min_val = eps;
%     X(X<=min_val) = min_val;
%     Pot = -log(X);
    
    % MDS
    Y = svdpca(pc_t, ndim, 'random');
    
    % plot
    figure;
    scatter(Y(:,1), Y(:,2), 5, C, 'filled');
    colormap(jet)
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    xlabel 'PC1'
    ylabel 'PC2'
    title(['PHATE: ' num2str(n_samp)])
    drawnow;
    
    time_vec(J) = toc
    
end


