%%
addpath(genpath('~/Documents/GitHub/MAGIC2/'));
addpath(genpath('~/Documents/GitHub/Blitz/'));

%% generate random fractal tree via DLA
rng(17) % 17
n_samp = 10000;
n_dim = 100;
n_branch = 50;
sigma = 5;
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

%% PCA
tic;
npca = 100;
[pc, ~, ~, q] = svdpca(M, npca, 'random');
toc
%figure;
%plot(q)

%% plot PCA 2D
figure;
scatter(pc(:,1), pc(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'

%% Construct diffusion operator
tic;
P = compute_operator2(pc, 'k', 25);
toc

%% fast pinv power
t = 24;
tic;
pc_t = pc;
for I=1:t
    I
    pc_t = P * pc_t;
end
P_t = pc_t * pinv(pc);
toc

%% potential
tic;
X = P_t;
min_val = eps;
X(X<=min_val) = min_val;
Pot = -log(X);
toc

%% fast CMDS
ndim = 2;
tic;
Y = svdpca(Pot, ndim, 'random');
toc

%% plot fast PHATE CMDS
figure;
scatter(Y(:,1), Y(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
title 'PHATE'

%% fast CMDS magic
ndim = 2;
tic;
Y_magic = svdpca(pc_t, ndim, 'random');
toc

%% plot fast PHATE CMDS
figure;
scatter(Y_magic(:,1), Y_magic(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
title 'MAGIC on PCA'










