%% init
cd('~/git_projects/Blitz/')
out_base = '~/Dropbox/Phate/Sherm_colon/figures/Dec7/'
mkdir(out_base);
addpath(genpath('~/git_projects/Blitz/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/Phate/Sherm_colon/';
sample_normal = 'Normal/';
sample_tumor = 'Tumor/';
sdata_normal = load_10xData([data_dir sample_normal]);
sdata_tumor = load_10xData([data_dir sample_tumor]);
sample_names = {'Normal', 'Tumor'};
sdata_raw = merge_data({sdata_normal, sdata_tumor}, sample_names);

%% to sdata
sdata = sdata_raw

%% library size hist
figure;
histogram(log10(sdata.library_size), 40);

%% lib size norm global
sdata = sdata.normalize_data_fix_zero();

%% sqrt transform
sdata.data = sqrt(sdata.data);

%% filter by hemoglobin (Hba-a1)
% x = get_channel_data(sdata, 'Hba-a1');
% figure;
% histogram(x);
% th = 7;
% cells_keep = x < th;
% sdata.data = sdata.data(cells_keep,:);
% sdata.cells = sdata.cells(cells_keep);
% sdata.library_size = sdata.library_size(cells_keep);
% sdata.samples = sdata.samples(cells_keep);
% x = get_channel_data(sdata, 'Hba-a1');
% figure;
% histogram(x);

%% remove empty genes
genes_keep = sum(sdata.data) > 0;
sdata.data = sdata.data(:,genes_keep);
sdata.genes = sdata.genes(genes_keep);
sdata.mpg = sdata.mpg(genes_keep);
sdata.cpg = sdata.cpg(genes_keep);
sdata = sdata.recompute_name_channel_map()

%% PCA
npca = 100;
pc = svdpca(sdata.data, npca, 'random');

%% plot PCA
figure;
c = sdata.samples;
scatter3(pc(:,1), pc(:,2), pc(:,3), 5, c, 'filled');
colormap(jet)
%colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
title 'PCA'
xlabel 'PC1'
ylabel 'PC2'
zlabel 'PC3'
view([100 15]);
h = colorbar;
ylabel(h, 'Sample');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_3D_samples.tiff']);
%close

%% MNN kernel
k = 3;
a = 15;
DiffOp = mnn_kernel(pc, sdata.samples, [], k, a);

%% normal kernel
% k = 3;
% a = 15;
% DiffOp = mnn_kernel(pc, [], [], k, a);

%% MAGIC
tic;
t = 6;
disp 'powering operator'
DiffOp_t = DiffOp^t;
sdata_imputed = sdata;
disp 'imputing'
sdata_imputed.data = DiffOp_t * sdata.data;
toc

%% PCA after MAGIC
npca = 100;
pc_magic = svdpca(sdata_imputed.data, npca, 'random');

%% plot PCA after MAGIC 3D
figure;
c = sdata.samples;
scatter3(pc_magic(:,1), pc_magic(:,2), pc_magic(:,3), 5, c, 'filled');
colormap(jet)
%colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
title 'MAGIC PCA'
xlabel 'PC1'
ylabel 'PC2'
zlabel 'PC3'
view([100 15]);
h = colorbar;
ylabel(h, 'Sample');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MAGIC_PCA_3D_samples.tiff']);
%close

%% interpolate time
t = 6;
t_vec = sdata.samples - 1;
for I=1:t
    I
    t_vec = DiffOp * t_vec;
end

%% MMDS 2D on MAGIC
X = pc_magic;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_magic_2D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS MAGIC 2D
figure;
c = sdata.samples;
%c = t_vec;
scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 5, c, 'filled');
colormap(jet)
%colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
%title 'MAGIC on PCA'
%xlabel 'PC1'
%ylabel 'PC2'
%h = colorbar;
%ylabel(h, 'Latent developmental time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_MAGIC_2D_samples.tiff']);
%close

%% plot MMDS MAGIC 2D
figure;
c = t_vec;
scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
% title 'MAGIC on PCA'
% xlabel 'PC1'
% ylabel 'PC2'
h = colorbar;
ylabel(h, 'Tumor-ness');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_MAGIC_2D_samples_imputed.tiff']);
%close

%% MMDS 2D on raw data
X = pc;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_raw_2D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS MAGIC 2D
figure;
c = sdata.samples;
%c = t_vec;
scatter(Y_mmds_raw_2D(:,1), Y_mmds_raw_2D(:,2), 5, c, 'filled');
colormap(jet)
%colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
%title 'MAGIC on PCA'
%xlabel 'PC1'
%ylabel 'PC2'
%h = colorbar;
%ylabel(h, 'Latent developmental time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_raw_2D_samples.tiff']);
%close

%% PHATE with Hellinger distance
t = 6;
distfun_mds = 'euclidean';
DiffOp_t = DiffOp^t;
disp 'potential recovery'
% DiffOp_t(DiffOp_t<=eps)=eps;
% DiffPot = -log(DiffOp_t);
DiffPot = sqrt(DiffOp_t); % Hellinger PHATE
npca = 100;
DiffPot_pca = svdpca(DiffPot, npca, 'random'); % to make pdist faster
D_DiffPot = squareform(pdist(DiffPot_pca, distfun_mds));

%% PHATE with log
% t = 6;
% distfun_mds = 'euclidean';
% DiffOp_t = DiffOp^t;
% disp 'potential recovery'
% DiffPot = -log(DiffOp_t + 1e-2);
% %DiffPot = -log(max(DiffOp_t, 1e-15));
% npca = 100;
% DiffPot_pca = svdpca(DiffPot, npca, 'random'); % to make pdist faster
% D_DiffPot = squareform(pdist(DiffPot_pca, distfun_mds));

%% PHATE with EMD
t = 6;
npca = 100;
DiffOp_t = DiffOp^t;
DiffOp_t_CDF = cumsum(DiffOp_t,2);
%DiffOp_t_CDF = svdpca(DiffOp_t_CDF, npca, 'random');
D_DiffPot = squareform(pdist(DiffOp_t_CDF, 'cityblock'));

%% CMDS PHATE
ndim = 10;
Y_phate_cmds = randmds(D_DiffPot, ndim);

%% Phate CMDS 3D
figure;
%c = sdata.samples;
c = t_vec;
scatter3(Y_phate_cmds(:,1), Y_phate_cmds(:,2), Y_phate_cmds(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'CMDS PHATE'
%xlabel 'MDS1'
%ylabel 'MDS2'
%zlabel 'MDS3'
view([100 15]);
%h = colorbar;
%ylabel(h, 'Time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'CMDS_PHATE_3D_samples.tiff']);
%close

%% Metric MDS PHATE 2D
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(D_DiffPot, ndim);
Y_phate_mmds = mdscale(D_DiffPot, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS PHATE 2D
figure;
c = t_vec;
scatter(Y_phate_mmds(:,1), Y_phate_mmds(:,2), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
title 'MMDS PHATE'
xlabel 'MDS1'
ylabel 'MDS2'
h = colorbar;
ylabel(h, 'Tumor score');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_PHATE_2D_samples_hellinhger_t12.tiff']);
%close

%% kNN DREMI of score vs all genes
num_bin = 20;
num_grid = 60;
gene_set = sdata_imputed.genes;
mi = nan(length(gene_set),1);
dremi = nan(length(gene_set),1);
for I=1:length(gene_set)
    I
    y = gene_set{I}
    [mi(I), dremi(I)] = dremi_knn(sdata_imputed, t_vec, y, 'num_grid', num_grid, 'num_bin', num_bin, 'k', 10, 'make_plots', false);
end

%% Heatmap of top DREMI genes clustered and sorted by tumor score
figure;
subplot(20,1,1:18.5);
N = 1000;
[~,sind_genes] = sort(dremi, 'descend');
[~,sind_cells] = sort(t_vec, 'ascend');
M = sdata_imputed.data(sind_cells,sind_genes(1:N));
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
imagesc(M');
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by tumor score)'
ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',[]);
subplot(20,1,20);
imagesc(t_vec(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Tumor score'
set(gcf,'paperposition',[0 0 16 12]);
print('-dtiff',[out_base 'heatmap_top_dremi_genes_tumor_score.tiff']);
%close

%% latent time vs expression subplot
genes = sdata_imputed.genes(sind_genes(1:24)); % plot top 24
genes = intersect(genes, upper(sdata_imputed.genes));
nr = floor(sqrt(length(genes)));
nc = ceil(length(genes) / nr);
figure;
for I=1:length(genes)
    I
    c = get_channel_data(sdata_imputed, genes{I});
    subplot(nr, nc, I);
    scatter(t_vec, c, 1, sdata.samples, 'filled');
    colormap(jet)
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    axis tight
    title(genes{I})
    if I==1
        xlabel 'Tumor score'
    end
    drawnow
end
set(gcf,'paperposition',[0 0 4*nc 3*nr]);
print('-dtiff',[out_base 'latent_time_vs_genes_samples.tiff']);
close

