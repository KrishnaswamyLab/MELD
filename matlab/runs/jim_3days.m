%% init
out_base = '~/Dropbox/noonan/figures/mar23_3_time_points/'
mkdir(out_base);
addpath(genpath('~/Documents/GitHub//Blitz/'));
addpath(genpath('~/Documents/GitHub/sparse-DREMI-project/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/noonan/data/';
sample_rm3 = 'rm3/'; % E14.5 WT Male
sample_c6p3 = 'P3E16WTM/'; % E16 WT Male
sample_p3 = 'p3/'; % E17.5 WT Male
sample_p5 = 'p5/'; % E17.5 WT Female
sample_p6 = 'p6/'; % E17.5 Het Male
sample_p2 = 'p2/'; % E17.5 Het Female
sdata_rm3 = load_10xData([data_dir sample_rm3]);
sdata_c6p3 = load_10xData([data_dir sample_c6p3]);
sdata_p3 = load_10xData([data_dir sample_p3]);
sdata_p6 = load_10xData([data_dir sample_p6]);
sdata_p5 = load_10xData([data_dir sample_p5]);
sdata_p2 = load_10xData([data_dir sample_p2]);

%% merge data
% sample_names = {'rm3', 'c6-p3', 'p3'};
% sdata_raw = merge_data({sdata_rm3, sdata_c6p3, sdata_p3}, sample_names);
% sample_names = {'p3', 'p6'};
% sdata_raw = merge_data({sdata_p3, sdata_p6}, sample_names);
sample_names = {'p3', 'p6', 'p5', 'p2'};
sdata_raw = merge_data({sdata_p3, sdata_p6, sdata_p5, sdata_p2}, sample_names);

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
x = get_channel_data(sdata, 'Hba-a1');
figure;
histogram(x);
th = 5;
cells_keep = x < th;
sdata.data = sdata.data(cells_keep,:);
sdata.cells = sdata.cells(cells_keep);
sdata.library_size = sdata.library_size(cells_keep);
sdata.samples = sdata.samples(cells_keep);
x = get_channel_data(sdata, 'Hba-a1');
figure;
histogram(x);

%% subsample
N = 7000;
cells_keep = randsample(size(sdata.data,1), N);
sdata.data = sdata.data(cells_keep,:);
sdata.cells = sdata.cells(cells_keep);
sdata.library_size = sdata.library_size(cells_keep);
sdata.samples = sdata.samples(cells_keep)

%% remove empty genes
genes_keep = sum(sdata.data) > 0;
sdata.data = sdata.data(:,genes_keep);
sdata.genes = sdata.genes(genes_keep);
sdata.ENSIDs = sdata.ENSIDs(genes_keep);
sdata.mpg = sdata.mpg(genes_keep);
sdata.cpg = sdata.cpg(genes_keep);
sdata = sdata.recompute_name_channel_map()

%% remove means per sample
% sdata_norm = sdata;
% sdata_norm.data = subtract_means(sdata.data, sdata.samples);

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
%title 'PCA'
%xlabel 'PC1'
%ylabel 'PC2'
%zlabel 'PC3'
view([100 15]);
%h = colorbar;
%ylabel(h, 'Time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_3D_samples.tiff']);
%close

%% MNN kernel
k = 10;
a = 15;
[DiffOp,K] = mnn_kernel_beta(pc, sdata.samples, [], k, a, 'cosine', 'cosine', 1, '.*');

%% MNN kernel
k = 5;
a = 15;
[DiffOp,K] = mnn_kernel_beta(pc, sdata.samples, [], k, a, 'euclidean', 'euclidean', 1, '+');

%% MAGIC
tic;
t = 12;
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
%c = t_vec;
scatter3(pc_magic(:,1), pc_magic(:,2), pc_magic(:,3), 5, c, 'filled');
colormap(jet)
%colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'PCA after MAGIC'
%xlabel 'PC1'
%ylabel 'PC2'
%zlabel 'PC3'
view([100 15]);
%h = colorbar;
%ylabel(h, 'Latent developmental time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_MAGIC_3D_samples.tiff']);
%close

%% interpolate time
t = 12;
t_vec_raw = sdata.samples - 1;
t_vec = t_vec_raw;
for I=1:t
    I
    t_vec = DiffOp * t_vec;
end
figure;
histogram(t_vec_raw)
figure;
histogram(t_vec)

%% impute wt het male female
t = 12;
disp 'powering operator'
DiffOp_t = DiffOp^t;
wthet_vec = sdata.samples;
femalemale_vec = sdata.samples;
wthet_vec(sdata.samples == 1 | sdata.samples == 3) = 0;
wthet_vec(sdata.samples == 2 | sdata.samples == 4) = 1;
femalemale_vec(sdata.samples == 3 | sdata.samples == 4) = 0;
femalemale_vec(sdata.samples == 1 | sdata.samples == 2) = 1;
wthet_vec_imputed = wthet_vec;
wthet_vec_imputed = DiffOp_t * wthet_vec_imputed;
femalemale_vec_imputed = femalemale_vec;
femalemale_vec_imputed = DiffOp_t * femalemale_vec_imputed;
corr_vec = femalemale_vec_imputed .* wthet_vec_imputed;

%%
figure;
scatter(femalemale_vec_imputed, wthet_vec_imputed, 5, corr_vec, 'filled');
xlabel 'female -> male'
ylabel 'wt -> het'

%% plot PCA after MAGIC 3D
figure;
c = corr_vec;
scatter3(pc_magic(:,1), pc_magic(:,2), pc_magic(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'PCA after MAGIC'
%xlabel 'PC1'
%ylabel 'PC2'
%zlabel 'PC3'
view([100 15]);
%h = colorbar;
%ylabel(h, 'Latent developmental time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_MAGIC_3D_t_vec_imputed.tiff']);
%close

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
%title 'MAGIC on PCA'
%xlabel 'PC1'
%ylabel 'PC2'
%h = colorbar;
%ylabel(h, 'Latent developmental time');
%set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_MAGIC_2D_samples_imputed.tiff']);
%close

%% latent time vs expression subplot, small
genes = {'CHD8' 'ANK2' 'CHD2' 'KATNAL2' 'WAC' 'SUV420H1'};
genes = intersect(genes, upper(sdata_imputed.genes));
nr = 3;
nc = ceil(length(genes) / nr);
figure;
for I=1:length(genes)
    I
    c = get_channel_data(sdata_imputed, genes{I});
    subplot(nr, nc, I);
    scatter(t_vec, c, 3, sdata.samples, 'filled');
    colormap(jet)
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    axis tight
    title(genes{I})
    drawnow
end
set(gcf,'paperposition',[0 0 4*nc 3*nr]);
print('-dtiff',[out_base 'latent_time_vs_genes_samples_small.tiff']);
close

%% original time vs expression subplot, few genes
genes = {'CHD8' 'ANK2' 'CHD2' 'KATNAL2' 'WAC' 'SUV420H1'};
genes = intersect(genes, upper(sdata_imputed.genes));
nr = 3;
nc = ceil(length(genes) / nr);
figure;
for I=1:length(genes)
    I
    c = get_channel_data(sdata, genes{I});
    subplot(nr, nc, I);
    scatter(t_vec_raw + randn(size(t_vec_raw))*0.15, c, 3, sdata.samples, 'filled');
    colormap(jet)
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    axis tight
    title(genes{I})
    drawnow
end
set(gcf,'paperposition',[0 0 4*nc 3*nr]);
print('-dtiff',[out_base 'original_time_vs_genes_samples_small.tiff']);
%close

%% original time vs expression subplot, few genes
genes = {'CHD8' 'SCN2A' 'ARID1B' 'NRXN1' 'SYNGAP1' 'DYRK1A' 'CHD2' 'ANK2' 'KDM5B' 'ADNP'};
genes = intersect(genes, upper(sdata_imputed.genes));
nr = floor(sqrt(length(genes)));
nc = ceil(length(genes) / nr);
figure;
for I=1:length(genes)
    I
    c = get_channel_data(sdata_imputed, genes{I});
    subplot(nr, nc, I);
    scatter(t_vec, c, 3, sdata.samples, 'filled');
    colormap(jet)
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    axis tight
    title(genes{I})
    if I==1
        xlabel 'latent developmental time'
    end
    drawnow
end
set(gcf,'paperposition',[0 0 4*nc 3*nr]);
print('-dtiff',[out_base 'latent_time_vs_genes_samples.tiff']);
close

%% subplot color by gene
genes = {'CHD8' 'SCN2A' 'ARID1B' 'NRXN1' 'SYNGAP1' 'DYRK1A' 'CHD2' 'ANK2' 'KDM5B' 'ADNP' 'POGZ' ...
    'SUV420H1' 'SHANK2' 'TBR1' 'GRIN2B' 'DSCAM' 'KMT2C' 'PTEN' 'SHANK3' 'TCF7L2' 'TRIP12' 'SETD5' ...
    'TNRC6B' 'ASH1L' 'CUL3' 'KATNAL2' 'WAC' 'NCKAP1'};
genes = intersect(genes, upper(sdata_imputed.genes));
nr = floor(sqrt(length(genes)));
nc = ceil(length(genes) / nr);
figure;
for I=1:length(genes)
    I
    c = get_channel_data(sdata_imputed, genes{I});
    subplot(nr, nc, I);
    scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 1, c, 'filled');
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    axis tight
    title(genes{I})
    axis off
    drawnow
end
set(gcf,'paperposition',[0 0 4*nc 3*nr]);
print('-dtiff',[out_base 'MMDS_MAGIC_2D_subplot_noax.tiff']);
close

%% kNN DREMI of score vs all genes
x = t_vec;
num_bin = 20;
num_grid = 60;
gene_set = sdata_imputed.genes;
mi = nan(length(gene_set),1);
dremi = nan(length(gene_set),1);
for I=1:length(gene_set)
    I
    y = gene_set{I}
    [mi(I), dremi(I)] = dremi_knn(sdata_imputed, x, y, 'num_grid', num_grid, 'num_bin', num_bin, 'k', 10, 'make_plots', false);
end

%% heatmap top dremi CDs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/gsea_cell_differentiation_markers.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
[~,sind_cells] = sort(t_vec);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'Cell differentiation markers'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by latent time)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(t_vec(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Latent time'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'heatmap_top_dremi_CD_genes.tiff']);
close

%% heatmap top dremi TFs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/TFs_AnimalTFDB.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
[~,sind_cells] = sort(t_vec);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'TFs (AnimalTFDB)'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by latent time)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(t_vec(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Latent time'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'heatmap_top_dremi_TFs.tiff']);
close

%% heatmap top dremi CMs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/ChromatinFactors_AnimalTFDB.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
[~,sind_cells] = sort(t_vec);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'Chromatin factors (AnimalTFDB)'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by latent time)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(t_vec(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Latent time'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'heatmap_top_dremi_CMs.tiff']);
%close

%% compute gene-gene corr
corr_mat = corr(sdata_imputed.data);
pc_genes_corr = svdpca(corr_mat, npca, 'random');
pc_genes_corr_abs = svdpca(abs(corr_mat), npca, 'random');

%% plot PCA on corr genes
figure;
c = log10(mean(sdata.data));
scatter3(pc_genes_corr(:,1), pc_genes_corr(:,2), pc_genes_corr(:,3), 5, c, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
title 'PCA on gene-gene corr'
xlabel 'PC1'
ylabel 'PC2'
zlabel 'PC3'
%view([100 15]);
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 16 12]);
print('-dtiff',[out_base 'PCA_corr_genes_3D.tiff']);
%close

%% plot PCA on corr genes abs
figure;
c = log10(mean(sdata.data));
scatter3(pc_genes_corr_abs(:,1), pc_genes_corr_abs(:,2), pc_genes_corr_abs(:,3), 5, c, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
title 'PCA on gene-gene abs corr'
xlabel 'PC1'
ylabel 'PC2'
zlabel 'PC3'
view([-70 30]);
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 16 12]);
print('-dtiff',[out_base 'PCA_abs_corr_genes_3D.tiff']);
%close

%% plot 2D PCA on corr genes
figure;
hold on;
c = log10(mean(sdata.data));
scatter(pc_genes_corr(:,1), pc_genes_corr(:,2), 5, c, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
title 'PCA on gene-gene corr'
xlabel 'PC1'
ylabel 'PC2'
idx_CHD8 = ismember(lower(sdata_imputed.genes), 'chd8');
scatter(pc_genes_corr(idx_CHD8,1), pc_genes_corr(idx_CHD8,2), 50, 'red', 'filled');
text(pc_genes_corr(idx_CHD8,1)+2, pc_genes_corr(idx_CHD8,2), 'CHD8');
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base '2D_PCA_corr_genes_CHD8.tiff']);
%close

%% plot 2D PCA on abs corr genes
figure;
hold on;
c = log10(mean(sdata.data));
scatter(pc_genes_corr_abs(:,1), pc_genes_corr_abs(:,2), 5, c, 'filled');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
title 'PCA on gene-gene corr'
xlabel 'PC1'
ylabel 'PC2'
idx_CHD8 = ismember(lower(sdata_imputed.genes), 'chd8');
scatter(pc_genes_corr_abs(idx_CHD8,1), pc_genes_corr_abs(idx_CHD8,2), 50, 'red', 'filled');
text(pc_genes_corr_abs(idx_CHD8,1)+2, pc_genes_corr_abs(idx_CHD8,2), 'CHD8');
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base '2D_PCA_abs_corr_genes_CHD8.tiff']);
%close

%% hellinger MDS on genes
data = sdata_imputed.data;
data = bsxfun(@rdivide, data, sum(data));
data = sqrt(data);
n_samp = randsample(size(data,2), 3000);

%% CMDS genes
ndim = 3;
pdx_genes = squareform(pdist(data(:,n_samp)', 'cosine'));
Y_genes_cmds = randmds(pdx_genes, ndim);

%% PCA genes
Y_genes_cmds = svdpca(data(:,n_samp)', 3, 'random');

%%


%% Phate CMDS 3D
figure;
c = log10(mean(sdata_imputed.data));
c = c(n_samp);
scatter3(Y_genes_cmds(:,1), Y_genes_cmds(:,2), Y_genes_cmds(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'CMDS PHATE'
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
%view([100 15]);
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'CMDS_genes_3D.tiff']);
%close

%% kNN DREMI of CHD8 vs all genes
x = get_channel_data(sdata_imputed, 'CHD8');
num_bin = 20;
num_grid = 60;
gene_set = sdata_imputed.genes;
dremi_chd8 = nan(length(gene_set),1);
for I=1:length(gene_set)
    I
    y = gene_set{I}
    [~, dremi_chd8(I)] = dremi_knn(sdata_imputed, x, y, 'num_grid', num_grid, 'num_bin', num_bin, 'k', 10, 'make_plots', false);
end

%% heatmap top dremi CDs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/gsea_cell_differentiation_markers.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
x = get_channel_data(sdata_imputed, 'CHD8');
[~,sind_cells] = sort(x);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'Cell differentiation markers, top MI with CHD8'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by Chd8 expression)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(x(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Chd8 expression'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'chd8_heatmap_top_dremi_CD_genes.tiff']);
close

%% heatmap top dremi TFs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/TFs_AnimalTFDB.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
x = get_channel_data(sdata_imputed, 'CHD8');
[~,sind_cells] = sort(x);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'TFs (AnimalTFDB), top MI with CHD8'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by Chd8 expression)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(x(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Chd8 expression'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'chd8_heatmap_top_dremi_TFs.tiff']);
close

%% heatmap top dremi CMs
genes_file = '~/Dropbox/EMT_dropseq/gene_lists/ChromatinFactors_AnimalTFDB.txt';
gene_set = read_gene_set(genes_file);
[~,LIA] = intersect(lower(sdata_imputed.genes), lower(gene_set));
N_top = 50;
[~,sind] = sort(dremi(LIA),'descend');
idx_top = LIA(sind(1:N_top));
genes = sdata_imputed.genes(idx_top);
x = get_channel_data(sdata_imputed, 'CHD8');
[~,sind_cells] = sort(x);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'Chromatin factors (AnimalTFDB), top MI with CHD8'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by Chd8 expression)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(x(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Chd8 expression'
set(gcf,'paperposition',[0 0 10 8]);
print('-dtiff',[out_base 'chd8_heatmap_top_dremi_CMs.tiff']);
close



%% MAGIC on gene space
n_samp = randsample(size(sdata_imputed.data,2), 5000);
data = sdata_imputed.data(:,n_samp);
data = bsxfun(@rdivide, data, sum(data));
data = sqrt(data);
data = abs(corr(data)); % gene-gene correlations
pc_genes = svdpca(data, 100, 'random');
%pc_genes = randmds(data', 100);
k = 7;
a = 10;
DiffOp_genes = mnn_kernel_beta(pc_genes, [], [], k, a, 'euclidean', 0.5);
data_imputed = DiffOp_genes^128 * data;
pc_genes_imputed = svdpca(data_imputed, 3, 'random');

%% plot imputed genes
figure;
c = log10(mean(sdata_imputed.data));
c = c(n_samp);
scatter3(pc_genes(:,1), pc_genes(:,2), pc_genes(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'CMDS PHATE'
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
%view([100 15]);
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_no_magic_genes_3D.tiff']);
%close

%% plot imputed genes
figure;
c = log10(mean(sdata_imputed.data));
c = c(n_samp);
scatter3(pc_genes_imputed(:,1), pc_genes_imputed(:,2), pc_genes_imputed(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
%title 'CMDS PHATE'
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
%view([100 15]);
h = colorbar;
ylabel(h, 'log10 mean expr.');
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_magic_genes_3D.tiff']);
%close

%% correlation on gene space, find genes close to CHD8
data = sdata_imputed.data;
data = bsxfun(@rdivide, data, sum(data));
data = sqrt(data);
data = abs(corr(data)); % gene-gene correlations
%%
idx_CHD8 = ismember(lower(sdata_imputed.genes), 'chd8');
corr_vec = data(idx_CHD8,:);
[q,sind] = sort(corr_vec, 'descend');
N_top = 100;
idx_top = sind(1:N_top);
genes = sdata_imputed.genes(idx_top);
x = get_channel_data(sdata_imputed, 'CHD8');
[~,sind_cells] = sort(x);
M = sdata_imputed.data(sind_cells,idx_top);
M = zscore(M);
D = pdist(M');
Z = linkage(D,'average');
order_genes = optimalleaforder(Z,D);
M = M(:,order_genes);
figure;
subplot(20,1,1:18);
imagesc(M');
title 'Top abs(corr) with CHD8'
set(gca,'clim',prctile(M(:), [0.5 99.5]));
xlabel 'Cells (sorted by Chd8 expression)'
%ylabel 'Genes'
set(gca,'xtick',[]);
set(gca,'ytick',1:size(M,1));
set(gca,'yticklabels',genes(order_genes));
subplot(20,1,20);
imagesc(x(sind_cells)');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
xlabel 'Chd8 expression'
set(gcf,'paperposition',[0 0 20 16]);
print('-dtiff',[out_base 'chd8_heatmap_top_abs_corr_chd8.tiff']);
close

