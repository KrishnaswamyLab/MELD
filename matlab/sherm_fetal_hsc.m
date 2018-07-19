%% init
cd('~/Documents/GitHub/Blitz/')
out_base = '~/Dropbox/Phate/Sherm_fetal_hsc/figures/Feb11_k7/'
mkdir(out_base);
addpath(genpath('~/Documents/GitHub/Blitz/'));
rseed = 7;

%% load data
data_dir = '~/Dropbox/Phate/Sherm_bm/';
sample_fetal = 'fetal/';
sdata_fetal = load_10xData([data_dir sample_fetal]);

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
npca = 100;
pc = svdpca(sdata.data, npca, 'random');

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
ylabel(h, 'Sample');
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'PCA_3D_samples.tiff']);
%close

%% Metric MDS raw 3D
X = pc;
X = squareform(pdist(X, 'euclidean'));
ndim = 3;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_3D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_mmds_3D.mat'], 'Y_mmds_3D');

%% plot MMDS raw 3D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter3(Y_mmds_3D(:,1), Y_mmds_3D(:,2), Y_mmds_3D(:,3), 10, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_raw_3D_samples_' gene '.tiff']);

%% Metric MDS raw 2D
X = pc;
X = squareform(pdist(X, 'euclidean'));
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_2D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_mmds_2D.mat'], 'Y_mmds_2D');

%% plot MMDS raw 2D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter(Y_mmds_2D(:,1), Y_mmds_2D(:,2), 5, c, 'filled');
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
print('-dtiff',[out_base 'MMDS_raw_2D_samples_' gene '.tiff']);
%close

%% operator
k = 7;
a = 15;
DiffOp = mnn_kernel_beta(pc, [], [], k, a, 'euclidean', 0.5);

%% optimal t
t_opt = compute_optimal_t(sdata.data, DiffOp, 't_max', 12, 'n_genes', 500, 'make_plots', true)

%% optimal t var
t_opt_var = compute_optimal_t_var(sdata.data, DiffOp, 't_max', 12, 'n_genes', 1, 'make_plots', true)

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

%% MMDS 3D on MAGIC
X = pc_magic;
X = squareform(pdist(X, 'euclidean'));
ndim = 3;
opt = statset('display', 'iter');
Y_start = randmds(X, ndim);
Y_mmds_magic_3D = mdscale(X, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_mmds_magic_3D.mat'], 'Y_mmds_magic_3D');

%% plot MMDS MAGIC 2D
figure;
gene = 'CD34';
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

%% plot MMDS MAGIC 3D
figure;
gene = 'Cd34';
c = get_channel_data(sdata_imputed, gene);
scatter3(Y_mmds_magic_3D(:,1), Y_mmds_magic_3D(:,2), Y_mmds_magic_3D(:,3), 10, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_MAGIC_3D_samples_' gene '.tiff']);
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
    scatter(Y_mmds_magic_2D(:,1), Y_mmds_magic_2D(:,2), 5, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c_raw>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_MAGIC_2D_key_genes.tiff']);
close

%% subplot key genes raw MMDS
gene_set = read_gene_set('~//Dropbox/Phate/Sherm_bm/key_genes.txt');
gene_set = intersect(lower(sdata.genes), lower(gene_set));
n_row = ceil(sqrt(length(gene_set)));
n_col = ceil(length(gene_set)/n_row);
figure;
for I=1:length(gene_set)
    subplot(n_row, n_col, I);
    c = get_channel_data(sdata_imputed, gene_set{I});
    c_raw = get_channel_data(sdata, gene_set{I});
    scatter(Y_mmds_2D(:,1), Y_mmds_2D(:,2), 5, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c_raw>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_raw_2D_key_genes.tiff']);
close

%% subplot key genes raw MMDS, no MAGIC
gene_set = read_gene_set('~//Dropbox/Phate/Sherm_bm/key_genes.txt');
gene_set = intersect(lower(sdata.genes), lower(gene_set));
n_row = ceil(sqrt(length(gene_set)));
n_col = ceil(length(gene_set)/n_row);
figure;
for I=1:length(gene_set)
    subplot(n_row, n_col, I);
    c = get_channel_data(sdata, gene_set{I});
    scatter(Y_mmds_2D(:,1), Y_mmds_2D(:,2), 5, c, 'filled');
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    axis tight
    title([gene_set{I} ' (' num2str(sum(c>0)) ')']);
end
set(gcf,'paperposition',[0 0 4*n_col 3*n_row]);
print('-dtiff',[out_base 'subplot_MMDS_raw_2D_key_genes_no_magic.tiff']);
close

%% VNE
tic;
[~,S,~] = randPCA(DiffOp, 5635);
toc;
S = diag(S);
%%
figure;
plot(S);
%%
t_max = 80;
H = nan(t_max,1);
for t=1:t_max
    S_t = S.^t;
    P = S_t ./ sum(S_t);
    H(t) = -sum(P(P>0) .* log(P(P>0)));
end
figure;
plot(1:t_max, H, '.-');

%% PHATE
t = 27;
distfun_mds = 'euclidean';
DiffOp_t = DiffOp^t;
disp 'potential recovery'
DiffOp_t(DiffOp_t<=eps)=eps;
DiffPot = -log(DiffOp_t);
%DiffPot = sqrt(DiffOp_t); % Hellinger PHATE
npca = 100;
DiffPot_pca = svdpca(DiffPot, npca, 'random'); % to make pdist faster
D_DiffPot = squareform(pdist(DiffPot_pca, distfun_mds));

%% CMDS PHATE
ndim = 10;
Y_phate_cmds = randmds(D_DiffPot, ndim);

%% plot CMDS PHATE 3D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter3(Y_phate_cmds(:,1), Y_phate_cmds(:,2), Y_phate_cmds(:,3), 5, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'CMDS_PHATE_3D_samples_' gene '.tiff']);
%close

%% Metric MDS PHATE 2D
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(D_DiffPot, ndim);
Y_phate_mmds_2D = mdscale(D_DiffPot, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_phate_mmds_2D.mat'], 'Y_phate_mmds_2D');

%% plot MMDS PHATE 2D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter(Y_phate_mmds_2D(:,1), Y_phate_mmds_2D(:,2), 5, c, 'filled');
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
print('-dtiff',[out_base 'MMDS_PHATE_2D_samples_' gene '.tiff']);
%close

%% non Metric MDS PHATE 2D
ndim = 2;
opt = statset('display', 'iter');
Y_start = randmds(D_DiffPot, ndim);
Y_phate_nmmds_2D = mdscale(D_DiffPot, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'stress');
save([out_base 'Y_phate_nmmds_2D.mat'], 'Y_phate_nmmds_2D');

%% plot NMMDS PHATE 2D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter(Y_phate_nmmds_2D(:,1), Y_phate_nmmds_2D(:,2), 5, c, 'filled');
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
print('-dtiff',[out_base 'NMMDS_PHATE_2D_samples_' gene '.tiff']);
%close

%% Metric MDS PHATE 3D
ndim = 3;
opt = statset('display', 'iter');
Y_start = randmds(D_DiffPot, ndim);
Y_phate_mmds_3D = mdscale(D_DiffPot, ndim, 'options', opt, 'start', Y_start, 'Criterion', 'metricstress');
save([out_base 'Y_phate_mmds_3D.mat'], 'Y_phate_mmds_3D');

%% plot MMDS PHATE 3D
figure;
gene = 'CD34';
c = get_channel_data(sdata_imputed, gene);
scatter3(Y_phate_mmds_3D(:,1), Y_phate_mmds_3D(:,2), Y_phate_mmds_3D(:,3), 10, c, 'filled');
colormap(parula)
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
axis tight
xlabel 'MDS1'
ylabel 'MDS2'
zlabel 'MDS3'
h = colorbar;
ylabel(h, gene);
set(h,'yticklabel',[]);
set(gcf,'paperposition',[0 0 8 6]);
print('-dtiff',[out_base 'MMDS_PHATE_3D_samples_' gene '.tiff']);
%close

%% write coordinates to csv
T = table();
% raw
T.MMDS_2D_1 = Y_mmds_2D(:,1);
T.MMDS_2D_2 = Y_mmds_2D(:,2);
T.blank1 = nan(size(Y_mmds_2D,1),1);
T.MMDS_3D_1 = Y_mmds_3D(:,1);
T.MMDS_3D_2 = Y_mmds_3D(:,2);
T.MMDS_3D_3 = Y_mmds_3D(:,3);
T.blank2 = nan(size(Y_mmds_2D,1),1);
% MAGIC
T.MMDS_2D_1_MAGIC = Y_mmds_magic_2D(:,1);
T.MMDS_2D_2_MAGIC = Y_mmds_magic_2D(:,2);
T.blank3 = nan(size(Y_mmds_2D,1),1);
T.MMDS_3D_1_MAGIC = Y_mmds_magic_3D(:,1);
T.MMDS_3D_2_MAGIC = Y_mmds_magic_3D(:,2);
T.MMDS_3D_3_MAGIC = Y_mmds_magic_3D(:,3);
T.blank4 = nan(size(Y_mmds_2D,1),1);
% PHATE
T.MMDS_2D_1_PHATE = Y_phate_mmds_2D(:,1);
T.MMDS_2D_2_PHATE = Y_phate_mmds_2D(:,2);
T.blank5 = nan(size(Y_mmds_2D,1),1);
T.MMDS_3D_1_PHATE = Y_phate_mmds_3D(:,1);
T.MMDS_3D_2_PHATE = Y_phate_mmds_3D(:,2);
T.MMDS_3D_3_PHATE = Y_phate_mmds_3D(:,3);
% % PHATE NMMDS
% T.NMMDS_2D_1_PHATE = Y_phate_nmmds_2D(:,1);
% T.NMMDS_2D_2_PHATE = Y_phate_nmmds_2D(:,2);
writetable(T,[out_base 'Fetal_HSC_coordinates.csv']);

%% write raw data to csv
fid = fopen([out_base 'data_raw.csv'],'w');
for I=1:length(sdata.genes)-1
    fprintf(fid,'%s,',sdata.genes{I});
end
fprintf(fid,'%s\n',sdata.genes{end});
for I=1:size(sdata.data,1)
    I
    fprintf(fid,[repmat('%6.2f,',1,size(sdata.data,2)-1) '%6.2f\n'],sdata.data(I,:));
end
fclose(fid);

%% write imputed data to csv
fid = fopen([out_base 'data_MAGIC.csv'],'w');
for I=1:length(sdata_imputed.genes)-1
    fprintf(fid,'%s,',sdata_imputed.genes{I});
end
fprintf(fid,'%s\n',sdata_imputed.genes{end});
for I=1:size(sdata_imputed.data,1)
    I
    fprintf(fid,[repmat('%6.2f,',1,size(sdata_imputed.data,2)-1) '%6.2f\n'],sdata_imputed.data(I,:));
end
fclose(fid);






