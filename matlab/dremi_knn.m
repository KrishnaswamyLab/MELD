function [mi, dremi] = ...
    dremi_knn(obj, channelx_name, channely_name, varargin)

%tic;

num_grid = 60;
num_bin = 20;
k = 10;
dist_fun = 'euclidean';
th = 0;
make_plots = false;
out_base = '';
prc = 0;
ms = 10;

for i=1:length(varargin)-1
    if(strcmp(varargin{i},'num_grid'))
        num_grid = varargin{i+1};
    end
    if(strcmp(varargin{i},'num_bin'))
        num_bin = varargin{i+1};
    end
    if(strcmp(varargin{i},'k'))
        k = varargin{i+1};
    end
    if(strcmp(varargin{i},'dist_fun'))
        dist_fun = varargin{i+1};
    end
    if(strcmp(varargin{i},'th'))
        th = varargin{i+1};
    end
    if(strcmp(varargin{i},'make_plots'))
        make_plots = varargin{i+1};
    end
    if(strcmp(varargin{i},'out_base'))
        out_base = varargin{i+1};
    end
    if(strcmp(varargin{i},'prc'))
        prc = varargin{i+1};
    end
end

% get channels
if ischar(channelx_name)
    channelx = obj.name_channel_map(channelx_name);
    channelx_data = obj.data(:, channelx);
else
    channelx_data = channelx_name;
    channelx_name = 'x';
end
if ischar(channely_name)
    channely = obj.name_channel_map(channely_name);
    channely_data = obj.data(:, channely);
else
    channely_data = channely_name;
    channely_name = 'y';
end
M = [channelx_data(:) channely_data(:)];

% zscore
M = zscore(M);

% remove outliers
if prc > 0
    px = prctile(M(:,1), [prc 100-prc]);
    py = prctile(M(:,2), [prc 100-prc]);
    indx = M(:,1) > px(1) & M(:,1) < px(2);
    indy = M(:,2) > py(1) & M(:,2) < py(2);
    M = M(indx & indy,:);
end

% bins
xl = [min(M(:,1)) max(M(:,1))];
yl = [min(M(:,2)) max(M(:,2))];
xedges_bin = linspace(xl(1), xl(2), num_bin+1);
yedges_bin = linspace(yl(1), yl(2), num_bin+1);
wx = (xl(2)-xl(1)) / (num_grid*2);
wy = (yl(2)-yl(1)) / (num_grid*2);
gridx = linspace(xl(1)+wx, xl(2)-wx, num_grid);
gridy = linspace(yl(1)+wy, yl(2)-wy, num_grid);
[gridx, gridy] = meshgrid(gridx, gridy);
gridx = gridx(:);
gridy = gridy(:);
gridxy = [gridx gridy];

% knn density per grid point to real points

if strcmp(k, 'auto')
    n = size(M,1);
    k = round(n^(-1/6) * sqrt(n)) % scot's rule times sqrt(n)
    %k = round(sqrt(n))
    [~, knn_dist] = knnsearch(M, gridxy, 'K', k+1, 'Distance', dist_fun);
    radius = knn_dist(:,k+1);
    volume = pi .* radius.^2;
    f = k ./ volume;
elseif strcmp(k, 'ensemble')
    n = size(M,1);
    k = 1:round(sqrt(n));
    f = nan(length(k),size(gridxy,1));
    for I=1:length(k)
        [~, knn_dist] = knnsearch(M, gridxy, 'K', k(I)+1, 'Distance', dist_fun);
        radius = knn_dist(:,k(I)+1);
        volume = pi .* radius.^2;
        f(I,:) = k(I) ./ volume;
    end
    w = 1./log2(k'+1);
    w = w ./ sum(w);
    disp(['mean k = ' num2str(sum(w .* k'))]);
    f = f .* repmat(w, 1, size(gridxy,1));
    f = sum(f);
elseif strcmp(k, 'ensemble2')
    n = size(M,1);
    k = 1:round(sqrt(n));
    w = 1./log2(k+1);
    w = w ./ sum(w);
    k = floor(sum(k .* w));
    disp(['k = ' num2str(k)]);
    [~, knn_dist] = knnsearch(M, gridxy, 'K', k+1, 'Distance', dist_fun);
    radius = knn_dist(:,k+1);
    volume = pi .* radius.^2;
    f = k ./ volume;
else
    [~, knn_dist] = knnsearch(M, gridxy, 'K', k+1, 'Distance', dist_fun);
    radius = knn_dist(:,k+1);
    volume = pi .* radius.^2;
    f = k ./ volume;
end
%volume = ((radius.^num_dims).*pi^(num_dims/2)) / gamma(1+(num_dims/2));

% bin grid points
[~,~,binx] = histcounts(gridxy(:,1),num_bin,'BinLimits',[xedges_bin(1),xedges_bin(end)]);
[~,~,biny] = histcounts(gridxy(:,2),num_bin,'BinLimits',[yedges_bin(1),yedges_bin(end)]);

%compute the coarse-grained joint distribution
joint_distro = zeros(num_bin, num_bin);
for I=1:length(f)
    joint_distro(biny(I), binx(I)) = joint_distro(biny(I), binx(I)) + f(I);
end
joint_distro = joint_distro ./ sum(sum(joint_distro));

%compute the conditional distribution and entropies
condition_entropies = nan(1, num_bin);
valid_bits = zeros(1, num_bin);

joint_distro_norm = zeros(num_bin, num_bin);
for i=1:num_bin
    %joint normalized to form conditional entropy
    if(sum(joint_distro(:,i)) > th)
        joint_distro_norm(:,i) = joint_distro(:,i) ./ sum(joint_distro(:,i));
        %entropy for each condition
        [condition_entropies(i), valid_bits(i)] = ordinary_entropy(joint_distro_norm(:,i));
    end
end

positive_indices = valid_bits > 0;

% MI
marginal_entropy = ordinary_entropy(sum(joint_distro,2));
condition_sums = sum(joint_distro);
conditional_entropy = sum(condition_entropies(positive_indices) .* condition_sums(positive_indices));
mi = marginal_entropy - conditional_entropy;

% DREMI
marginal_entropy_norm = ordinary_entropy(sum(joint_distro_norm,2));
condition_sums_norm = mean(joint_distro_norm);
conditional_entropy_norm = sum(condition_entropies(positive_indices) .* condition_sums_norm(positive_indices));
dremi = marginal_entropy_norm - conditional_entropy_norm;

%t = toc;
%disp(['knn DREMI took ' num2str(t,2) ' seconds to compute']);

% plot

if make_plots
    
    figure;
    subplot(2,3,1);
    dscatter(M(:,1), M(:,2));
    colormap(hot);
    xlim(xl);
    ylim(yl);
    xlabel(channelx_name);
    ylabel(channely_name);
    title 'density scatter'
    
    subplot(2,3,2);
    hold all
    c = log(f);
    scatter(gridx, gridy, ms, c, 'filled');
    scatter(M(:,1), M(:,2), 2, 'k', 'filled');
    colormap(hot);
    xlim(xl);
    ylim(yl);
    xlabel(channelx_name);
    ylabel(channely_name);
    title 'grid points density (log)'
    
    subplot(2,3,3);
    hold all
    c = log(f);
    scatter(gridx, gridy, ms, c, 'filled');
    colormap(hot);
    for i=1:length(xedges_bin)
        plot([xedges_bin(i) xedges_bin(i)], [min(yedges_bin) max(yedges_bin)], '-k');
    end
    for i=1:length(yedges_bin)
        plot([min(xedges_bin) max(xedges_bin)], [yedges_bin(i) yedges_bin(i)], '-k');
    end
    xlim(xl);
    ylim(yl);
    xlabel(channelx_name);
    ylabel(channely_name);
    title 'grid points density (log) + bins'
    
    subplot(2,3,4);
    hold all
    dscatter(M(:,1), M(:,2));
    colormap(hot);
    for i=1:length(xedges_bin)
        plot([xedges_bin(i) xedges_bin(i)], [min(yedges_bin) max(yedges_bin)], '-k');
    end
    for i=1:length(yedges_bin)
        plot([min(xedges_bin) max(xedges_bin)], [yedges_bin(i) yedges_bin(i)], '-k');
    end
    xlim(xl);
    ylim(yl);
    xlabel(channelx_name);
    ylabel(channely_name);
    title 'density scatter + bins'
    
    subplot(2,3,5);
    imagesc(joint_distro);
    colormap(hot);
    set(gca,'ydir','normal');
    title 'joint distro'
    xlabel(channelx_name)
    ylabel(channely_name)
    
    subplot(2,3,6);
    imagesc(joint_distro_norm);
    colormap(hot);
    set(gca,'ydir','normal');
    title(['joint distro norm, mi = ' num2str(mi,3) ', dremi = ' num2str(dremi,3)]);
    xlabel(channelx_name)
    ylabel(channely_name)
    
    set(gcf,'paperposition',[0 0 18 10]);
    print('-dtiff',[out_base 'knn_dremi_' channelx_name '_vs_' channely_name '_ng_' num2str(num_grid) '_nb_' num2str(num_bin) '.tiff']);
    
end

