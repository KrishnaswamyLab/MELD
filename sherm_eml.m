%% init
cd('~/Documents/GitHub/Blitz/')
out_base = '~/Dropbox/Phate/Sherm_eml/figures/mar11/'
mkdir(out_base);
addpath(genpath('~/Documents/GitHub/Blitz/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/Phate/';
sample1 = 'Sherm_eml/';
sdata_fetal = load_10xData([data_dir sample1]);

%% to sdata
sdata = sdata_fetal

%% random sample
% N = 3000;
% cells_keep = randsample(size(sdata.data,1), N);
% sdata.data = sdata.data(cells_keep,:);
% sdata.cells = sdata.cells(cells_keep);
% sdata.library_size = sdata.library_size(cells_keep);
% %sdata.samples = sdata.samples(cells_keep);

%% remove empty genes
genes_keep = sum(sdata.data) > 0;
sdata.data = sdata.data(:,genes_keep);
sdata.genes = sdata.genes(genes_keep);
sdata.mpg = sdata.mpg(genes_keep);
sdata.cpg = sdata.cpg(genes_keep);
sdata = sdata.recompute_name_channel_map()

%% library size hist
figure;
histogram(log10(sdata.library_size), 40);

%% lib size norm global
sdata = sdata.normalize_data_fix_zero();

%% sqrt transform
sdata.data = sqrt(sdata.data);

%% PCA
tic;
[pc,U,mu,S] = svdpca(sdata.data, 50, 'random');
toc

%% singular values
figure;
plot(S, 'x-');

%% plot PCA
figure;
c = get_channel_data(sdata, 'CD34');
scatter3(pc(:,1), pc(:,2), pc(:,3), 5, c, 'filled');
%colormap(jet)
colormap(parula)
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
ylabel(h, 'CD34');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_3D_samples.tiff']);
%close

%% operator
k = 3;
a = 15;
DiffOp = mnn_kernel_beta(pc, [], [], k, a, 'euclidean', 0.5);

%% optimal t
t_opt = compute_optimal_t(sdata.data, DiffOp, 't_max', 12, 'n_genes', 500, 'make_plots', true)

%% MAGIC
tic;
t = 24;
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
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter3(pc_magic(:,1), pc_magic(:,2), pc_magic(:,3), 5, c, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
zlabel 'PC3'
view([100 15]);
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MAGIC_PCA_3D_samples_' gene '.tiff']);
%close

%% MMDS 2D on MAGIC
X = pc_magic;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_magic_2D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_mmds_magic_2D.mat'], 'Y_mmds_magic_2D');

%% plot MMDS MAGIC 2D
figure;
gene = 'Tcf7';
c = get_channel_data(sdata_imputed, gene);
scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'MDS1'
ylabel 'MDS2'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_MAGIC_2D_samples_' gene '.tiff']);
%close

%% subplot key genes MAGIC MMDS
gene_set = read_gene_set('~//Dropbox/Phate/Sherm_bm/key_genes.txt');
gene_set = intersect(lower(sdata.genes), lower(gene_set));
n_row = ceil(sqrt(length(gene_set)));
n_col = ceil(length(gene_set)/n_row);
figure;
for I=1:length(gene_set)
    subplot(n_row, n_col, I);
    c = get_channel_data(sdata_imputed, gene_set{I});
    c_raw = get_channel_data(sdata, gene_set{I});
    scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 1, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c_raw>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_MAGIC_2D_key_genes.tiff']);
close

%% subplot key genes MAGIC MMDS
gene_set = read_gene_set('~//Dropbox/Phate/Sherm_bm/key_genes.txt');
gene_set = intersect(lower(sdata.genes), lower(gene_set));
n_row = ceil(sqrt(length(gene_set)));
n_col = ceil(length(gene_set)/n_row);
figure;
for I=1:length(gene_set)
    subplot(n_row, n_col, I);
    c = get_channel_data(sdata, gene_set{I});
    c_raw = get_channel_data(sdata, gene_set{I});
    scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 1, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c_raw>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_MAGIC_2D_key_genes_raw.tiff']);
close


%% subplot key genes MAGIC MMDS
gene_set = read_gene_set('~//Dropbox/Phate/Sherm_bm/key_genes.txt');
gene_set = intersect(lower(sdata.genes), lower(gene_set));
n_row = ceil(sqrt(length(gene_set)));
n_col = ceil(length(gene_set)/n_row);
figure;
for I=1:length(gene_set)
    subplot(n_row, n_col, I);
    c = get_channel_data(sdata, gene_set{I}) > 0;
    c_raw = get_channel_data(sdata, gene_set{I});
    scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 1, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c_raw>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_MAGIC_2D_key_genes_raw_zero.tiff']);
close





