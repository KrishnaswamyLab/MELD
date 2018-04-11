function [DiffOp, K] = mnn_kernel_sparse(data, sample_ind, npca, k, distfun, beta, kernel_symm, gamma)

if isempty(sample_ind)
    sample_ind = ones(size(data,1),1); % sample_ind identifies each of multiple concatenated samples in a single matrix
end

uniq_samp = unique(sample_ind);
n_samp = length(uniq_samp);

% adaptive k based on sample sizes
n_cells_vec = nan(1,n_samp);
for I=1:n_samp
    n_cells_vec(I) = sum(sample_ind == I);
end
n_cells_vec
n_cells_weight = n_cells_vec / min(n_cells_vec)
k_mat = k * repmat(n_cells_weight,n_samp,1)
k_mat = round(k_mat)

if npca > 0
    M = svdpca(data, npca, 'random');
else
    M = data;
end

K = sparse(size(data,1));
for I=1:n_samp
    I
    samp_I = uniq_samp(I);        % sample index at position I
    idx_I = sample_ind == samp_I; % logical index for obs with samp_ind
    MI = M(idx_I,:);              % slice reduce data matrix for in sample_I points
    for J=1:n_samp
        samp_J = uniq_samp(J);
        idx_J = sample_ind == samp_J;
        if I==J
            KIJ = kernel_sparse(MI, MI, k+1, distfun) * beta;   % kernel between points in sample_I
        else
            MJ = M(idx_J,:);                                    % slice reduce data matrix for in sample_J points
            KIJ = kernel_sparse(MI, MJ, k, distfun);            % kernel from each point in sample_I to each in sample_J
        end
        K(idx_I, idx_J) = KIJ;  % fill out values in K for NN from I -> J
    end
end

switch kernel_symm
    case '+'
        K = K + K';
    case '*'
        K = K * K';
    case '.*'
        K = K .* K';
    case 'gamma'
        K = gamma * min(K,K') + (1-gamma) * max(K,K');
    otherwise
        disp('Unknown kernel symmetrization')
end

% num_cells = size(K,1);
% idx_lonely = full(sum(K) < k_min);
% K = K(~idx_lonely,~idx_lonely);
% idx_not_lonely = ~idx_lonely;
% num_lonely = sum(idx_lonely);
% if num_lonely > 0
%     disp(['!! Warning: Removed ' num2str(num_lonely) ' disconnected points out of ' num2str(num_cells) ' total !!'])
% end

disp 'computing operator'
DiffOp = bsxfun(@rdivide, K, sum(K,2));

disp 'done'
