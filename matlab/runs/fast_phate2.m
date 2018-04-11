%%
addpath(genpath('~/Documents/GitHub/MAGIC2/'));
addpath(genpath('~/Documents/GitHub/Blitz/'));

%% generate random fractal tree via DLA
rng(17) % 17
n_samp = 10000;
n_dim = 100;
n_branch = 40;
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

%%
[Y_pc,~,S] = svdpca(M, 100, 'random');
figure;
plot(S);
xlabel 'PC components'
ylabel 'Eigenvalue'

%% PCA before PHATE
[Y_pc,~,S] = svdpca(M, 3, 'random');

% plot PCA 2D
figure;
scatter(Y_pc(:,1), Y_pc(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
title 'PCA'

% plot PCA 3D 
figure;
scatter3(Y_pc(:,1), Y_pc(:,2), Y_pc(:,3), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
ylabel 'PC3'
title 'PCA'

%%
rng(7);
t = 48;
npca = 100; % for kernel
nsvd = 100; % for spectral clustering
ncluster = 1000;
ndim = 3;
k = 25;
pot_method = 'log';

tic;

pc = svdpca(M, npca, 'random');

% Construct diffusion operator
P = compute_operator2(pc, 'k', k); % can be made faster with kdtree or if ncol of pc is at most 10

% spectral cluster
[U,S,~] = randPCA(P, nsvd);
IDX = kmeans(U*S, ncluster);

% create Pnm and Pmn
n = size(P,1);
m = max(IDX);
Pnm = nan(n,m);
Pmn = nan(m,n);
for I=1:m
    Pnm(:,I) = sum(P(:,IDX==I),2);
    Pmn(I,:) = sum(P(IDX==I,:),1);
end
Pmn = bsxfun(@rdivide, Pmn, sum(Pmn,2));
   
% diffuse
P_t = Pnm * (Pmn * Pnm)^t;

% potential
switch pot_method
    case 'log'
        X = P_t;
        X(X<=eps) = eps;
        Pot = -log(X);
    case 'sqrt'
        Pot = sqrt(P_t);
    otherwise
        disp 'potential method not known'
end

% fast CMDS
Y = svdpca(Pot, ndim, 'random');

toc

%% plot

% plot fast PHATE CMDS
figure;
scatter(Y(:,1), Y(:,2), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
title 'PHATE'

% plot fast PHATE CMDS 3D
figure;
scatter3(Y(:,1), Y(:,2), Y(:,3), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
ylabel 'PC3'
title 'PHATE'

%% fast MMDS
rng(7);
t = 48;
npca = 100; % for kernel
nsvd = 100; % for spectral clustering
ncluster = 1000;
ndim = 3;
k = 25;
pot_method = 'log';

tic;

pc = svdpca(M, npca, 'random');

% Construct diffusion operator
P = compute_operator2(pc, 'k', k); % can be made faster with kdtree or if ncol of pc is at most 10

% spectral cluster
[U,S,~] = randPCA(P, nsvd);
IDX = kmeans(U*S, ncluster);

% create Pnm and Pmn
n = size(P,1);
m = max(IDX);
Pnm = nan(n,m);
Pmn = nan(m,n);
for I=1:m
    Pnm(:,I) = sum(P(:,IDX==I),2);
    Pmn(I,:) = sum(P(IDX==I,:),1);
end
Pmn = bsxfun(@rdivide, Pmn, sum(Pmn,2));

% diffuse
Pmm_t = (Pmn * Pnm)^t;

% potential
switch pot_method
    case 'log'
        X = Pmm_t;
        X(X<=eps) = eps;
        Pot = -log(X);
    case 'sqrt'
        Pot = sqrt(Pmm_t);
    otherwise
        disp 'potential method not known'
end

PDX = squareform(pdist(Pot));

opt = statset('display','iter');
Y_start = randmds(PDX,ndim);
Y_mmds = mdscale(PDX,ndim,'options',opt,'start',Y_start,'Criterion','metricstress');

Y_mmds = Pnm * Y_mmds;

toc

%% plot 2D

% plot fast PHATE CMDS
figure;
scatter(Y_mmds(:,1), Y_mmds(:,2), 5, C, 'filled');
colormap(hsv)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
title 'PHATE MMDS'

%% plot 3D

% plot fast PHATE CMDS 3D
figure;
scatter3(Y_mmds(:,1), Y_mmds(:,2), Y_mmds(:,3), 5, C, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
ylabel 'PC3'
title 'PHATE MMDS'



