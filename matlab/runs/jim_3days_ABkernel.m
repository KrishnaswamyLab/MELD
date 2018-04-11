%% init
out_base = '~/Dropbox/noonan/figures/mar22_3_time_points/'
mkdir(out_base);
addpath(genpath('~/Documents/GitHub//Blitz/'));
addpath(genpath('~/Documents/GitHub/sparse-DREMI-project/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/noonan/data/';
sample_rm3 = 'rm3/'; % E14.5 WT Male
sample_c6p3 = 'P3E16WTM/'; % E16 WT Male
%sample_p3 = 'p3/'; % E17.5 WT Male
sdata_rm3 = load_10xData([data_dir sample_rm3]);
sdata_c6p3 = load_10xData([data_dir sample_c6p3]);
%sdata_p3 = load_10xData([data_dir sample_p3]);

%% merge data
%sample_names = {'rm3', 'c6-p3', 'p3'};
%sdata_raw = merge_data({sdata_rm3, sdata_c6p3, sdata_p3}, sample_names);
sample_names = {'rm3', 'p3'};
sdata_raw = merge_data({sdata_rm3, sdata_p3}, sample_names);

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
N = 5000;
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

%% plot PCA 2D
figure;
c = sdata.samples;
scatter(pc(:,1), pc(:,2), 5, c, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_2D_samples.tiff']);
%close

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

%% batch correct data
t = 32;
k = 40;
a = 15;
npca = 100;
sdata_corr = sdata;
sdata_corr.data(sdata.samples==1,:) = batch_kernel(sdata.data(sdata.samples==1,:), ...
    sdata.data(sdata.samples==2,:), npca, k, a, t, 'euclidean');
sdata_corr.data(sdata.samples==2,:) = batch_kernel(sdata.data(sdata.samples==2,:), ...
    sdata.data(sdata.samples==1,:), npca, k, a, t, 'euclidean');

%%
sdata_corr = sdata;
mu_vec1 = mean(sdata.data(sdata.samples==1,:));
mu_vec2 = mean(sdata.data(sdata.samples==2,:));
sdata_corr.data(sdata.samples==1,:) = bsxfun(@minus, sdata_corr.data(sdata.samples==1,:), mu_vec1);
sdata_corr.data(sdata.samples==2,:) = bsxfun(@minus, sdata_corr.data(sdata.samples==2,:), mu_vec2);

%% PCA
npca = 100;
pc_corr = svdpca(sdata_corr.data, npca, 'random');

%% plot PCA 2D
figure;
c = sdata_corr.samples;
scatter(pc_corr(:,1), pc_corr(:,2), 5, c, 'filled');
colormap(jet)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
xlabel 'PC1'
ylabel 'PC2'
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_corr_2D_samples.tiff']);
%close

%% plot PCA
figure;
c = sdata_corr.samples;
scatter3(pc_corr(:,1), pc_corr(:,2), pc_corr(:,3), 5, c, 'filled');
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
k = 40;
a = 15;
[DiffOp,K] = mnn_kernel_beta(pc, sdata.samples, [], k, a, 'euclidean', 0.5);

%% MAGIC
tic;
t = 40;
disp 'powering operator'
DiffOp_t = DiffOp^t;
sdata_imputed = sdata_corr;
disp 'imputing'
sdata_imputed.data = DiffOp_t * sdata_corr.data;
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

