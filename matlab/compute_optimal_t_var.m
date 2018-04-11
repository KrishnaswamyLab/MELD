function [t_opt, H_vec] = compute_optimal_t_var(data, DiffOp, varargin)
% [t_opt, H_vec] = compute_optimal_t_var(data, DiffOp, varargin)
%   data - input data
%   DiffOp - diffusion operator
%   varargin:
%       t_max - max t to try
%       n_genes - number of random genes to compute optimal t on, should be
%        at least 100, fewer is faster
%       make_plots - draw convergence as a function of t with which we
%        select the optimal t

t_max = 32;
n_genes = size(data,2);
make_plots = true;

if ~isempty(varargin)
    for j = 1:length(varargin)
        if strcmp(varargin{j}, 't_max')
            t_max = varargin{j+1};
        end
        if strcmp(varargin{j}, 'n_genes')
            n_genes = varargin{j+1};
        end
        if strcmp(varargin{j}, 'make_plots')
            make_plots = varargin{j+1};
        end
    end
end

if ~issparse(DiffOp)
    DiffOp = sparse(DiffOp);
end

idx_genes = randsample(size(data,2), n_genes);
data_imputed = data;
data_imputed = data_imputed(:,idx_genes);

if min(data_imputed(:)) < 0
    disp 'data has negative values, shifting to positive'
    data_imputed = data_imputed - min(data_imputed(:));
end

H_vec = nan(t_max,1);
disp 'computing optimal t'
n_cells = size(data_imputed,1);
p_vec = ones(n_cells,1)./n_cells;
max_entr = -sum(p_vec .* log(p_vec));
p_mat = bsxfun(@rdivide, data_imputed, sum(data_imputed));
p_mat(p_mat==0) = eps;
H_prev = mean(-sum(p_mat .* log(p_mat),1),2);
for I=1:t_max
    disp(['t = ' num2str(I)]);
    data_imputed = DiffOp * data_imputed;
    p_mat = bsxfun(@rdivide, data_imputed, sum(data_imputed)) + eps;
    p_mat(p_mat==0) = eps;
    H_curr = mean(-sum(p_mat .* log(p_mat),1),2);
    H_vec(I) = (H_curr - H_prev) ./ max_entr;
    H_prev = H_curr;
end


t_opt = find(H_vec < 0.01, 1, 'first') + 1;
disp(['optimal t = ' num2str(t_opt)]);

if make_plots
    figure;
    hold all;
    plot(1:t_max, H_vec, '*-');
    plot(t_opt, H_vec(t_opt), 'or', 'markersize', 10);
    xlabel 't'
    ylabel 'H'
    axis tight
    %ylim([0 1]);
    plot(xlim, [0.01 0.01], '--k');
    legend({'y' 'optimal t' 'y=0.1'});
    set(gca,'xtick',1:t_max);
    set(gca,'xticklabel',1:t_max);
    set(gca,'ytick',0:0.1:1);
    
%     figure;
%     hold all;
%     plot(log10(1:t_max), log10(H_vec), '*-');
%     xlabel 'log10(t)'
%     ylabel 'log10(H)'
%     set(gca,'xtick',log10(1:t_max));
%     set(gca,'xticklabel',1:t_max);
%     axis tight
    
%     figure;
%     hold all;
%     plot(1:t_max+1, log10(H_vec), '*-');
%     xlabel 't'
%     ylabel 'log10 H'
%     axis tight
%     
%     figure;
%     hold all;
%     plot(log10(1:t_max+1), H_vec, '*-');
%     xlabel 'log10 t'
%     ylabel 'H'
%     axis tight
end


