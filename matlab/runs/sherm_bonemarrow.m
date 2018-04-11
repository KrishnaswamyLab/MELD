%% init
cd('~/Documents/GitHub/Blitz/')
out_base = '~/Dropbox/Phate/Sherm_bm/figures/Jan12/'
mkdir(out_base);
addpath(genpath('~/Documents/GitHub/Blitz/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/Phate/Sherm_bm/';
sample_p9 = 'P9/';
sample_p10 = 'P10/';
sample_p11 = 'P11/';
sample_p12 = 'P12/';
sample_p13 = 'P13/';
sample_p14 = 'P14/';
sdata_p9 = load_10xData([data_dir sample_p9]);
sdata_p10 = load_10xData([data_dir sample_p10]);
sdata_p11 = load_10xData([data_dir sample_p11]);
sdata_p12 = load_10xData([data_dir sample_p12]);
sdata_p13 = load_10xData([data_dir sample_p13]);
sdata_p14 = load_10xData([data_dir sample_p14]);
sample_names = {'SF', 'MPP4', 'MPP3', 'MPP1', 'HSC', 'MPP2'};
sdata_raw = merge_data({sdata_p9, sdata_p10, sdata_p11, sdata_p12, sdata_p13, sdata_p14}, sample_names);

%% to sdata
sdata = sdata_raw

%% just MPP2
sdata = sdata_p14

%% random sample
N = 3000;
cells_keep = randsample(size(sdata.data,1), N);
sdata.data = sdata.data(cells_keep,:);
sdata.cells = sdata.cells(cells_keep);
sdata.library_size = sdata.library_size(cells_keep);
%sdata.samples = sdata.samples(cells_keep);

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
npca = 100;
pc = svdpca(sdata.data, npca, 'random');

%% plot PCA
figure;
c = sdata.samples;
c = 'k'
c = get_channel_data(sdata_imputed, 'CD34');
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
k = 3;
a = 15;
DiffOp = mnn_kernel(pc, [], [], k, a);

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
c = 'k'
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

%% plot PCA after MAGIC 2D gscatter
figure;
nplot = 7;
K = 1;
usamp = unique(sdata.samples);
for I=1:nplot
    I
    for J=1:nplot
        subplot(nplot,nplot,K);
        K = K + 1;
        if I==J
            hold all;
            for L=1:length(usamp)
                curr_ind = ismember(sdata.samples, usamp(L));
                histogram(pc_magic(curr_ind,I));
            end
        else
            gscatter(pc_magic(:,I), pc_magic(:,J), sdata.samples, [], [], 1);
            if I==1 && J==2
                legend(sample_names);
                legend('location','best');
            else
                legend off;
            end
        end
        set(gca,'xticklabel',[]);
        set(gca,'yticklabel',[]);
        set(gca,'zticklabel',[]);
        axis tight
        if I == nplot
            xlabel(['PCA' num2str(J)]);
        end
        if J == 1 && I>1
            ylabel(['PCA' num2str(I)]);
        end
        if I == J
            xlabel(['PCA' num2str(I)]);
        end
    end
end
set(gcf,'paperposition',[0 0 nplot*6 nplot*5]);
print('-dtiff',[out_base 'MAGIC_PCA_2D_samples_gscatter_subplot.tiff']);
close

%% MMDS 2D on MAGIC
X = pc_magic;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_magic_2D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS MAGIC 2D
figure;
gscatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), sdata.samples, [], ['x' 'o' 'v' '^' 's'], 5);
legend('location','NW')
legend(sample_names);
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
close

%% plot MMDS MAGIC 2D
figure;
%c = sdata.samples;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
axis tight
title 'MMDS MAGIC'
xlabel 'MDS1'
ylabel 'MDS2'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
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

%% CMDS PHATE
ndim = 10;
Y_phate_cmds = randmds(D_DiffPot, ndim);

%% Phate CMDS 3D
figure;
c = sdata.samples;
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
c = sdata.samples;
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
print('-dtiff',[out_base 'MMDS_PHATE_2D_samples_hellinhger.tiff']);
%close

%% Metric MDS raw 2D
X = pc;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS raw 2D
figure;
%c = sdata.samples;
c = get_channel_data(sdata_imputed, 'CD34');
scatter(Y_mmds(:,1), Y_mmds(:,2), 5, c, 'filled');
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
print('-dtiff',[out_base 'MMDS_raw_2D_samples.tiff']);
%close

%% Metric MDS raw 3D
X = pc;
X = squareform(pdist(X, 'euclidean'));
ndim = 3;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_3D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');

%% plot MMDS raw 3D
figure;
%c = sdata.samples;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter3(Y_mmds_3D(:,1), Y_mmds_3D(:,2), Y_mmds_3D(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
title 'MMDS PHATE'
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS2'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_raw_3D_samples.tiff']);
%close

