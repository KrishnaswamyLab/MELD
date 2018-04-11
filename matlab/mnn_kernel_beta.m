function [DiffOp, K] = mnn_kernel_beta(data, sample_ind, npca, k, a, distfun_II, distfun_IJ, beta, kernel_symm)
% [DiffOp, K] = mnn_kernel(data, sample_ind, npca, k, a, distfun, beta)
%
%   creates a kernel that in combination with MAGIC does batch correction
%
%   sample_ind are the sample indices
%
%   k is k for the sample with fewest numbner of cells, k for othe samples
%   are weighted by cell number

if isempty(sample_ind)
    sample_ind = ones(size(data,1),1); % sample_ind identifies each of multiple concatenated samples in a single matrix
end

uniq_samp = unique(sample_ind);
n_samp = length(uniq_samp);

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

K = nan(size(data,1));
for I=1:n_samp
    I
    samp_I = uniq_samp(I);        % sample index at position I
    idx_I = sample_ind == samp_I; % logical index for obs with samp_ind
    MI = M(idx_I,:);              % slice reduce data matrix for in sample_I points
    for J=1:I
        samp_J = uniq_samp(J);
        idx_J = sample_ind == samp_J;
        MJ = M(idx_J,:);                 % slice reduce data matrix for in sample_J points
        if I==J
            PDXIJ = pdist2(MI, MJ, distfun_II); % distance between points in sample_I
        else
            PDXIJ = pdist2(MI, MJ, distfun_IJ); % distance from each point in sample_I to each in sample_J
        end
        knnDSTIJ = sort(PDXIJ,2);        % get KNN
        epsilonIJ = knnDSTIJ(:,k_mat(I,J));       % distance to KNN
        PDXIJ = bsxfun(@rdivide,PDXIJ,epsilonIJ); % normalize PDXIJ
        KIJ = exp(-PDXIJ.^a);            % apply alpha-decaying kernel
        K(idx_I, idx_J) = KIJ;  % fill out values in K for NN from I -> J
        if I~=J
            PDXJI = PDXIJ';         % Repeat to find KNN from J -> I
            knnDSTJI = sort(PDXJI,2);
            epsilonJI = knnDSTJI(:,k_mat(I,J));
            PDXJI = bsxfun(@rdivide,PDXJI,epsilonJI);
            KJI = exp(-PDXJI.^a);
            K(idx_J, idx_I) = KJI;
        end
        if I==J
            K(idx_I, idx_J) = K(idx_I, idx_J) * beta;
        else
%             K(idx_I, idx_J) = K(idx_I, idx_J) * ((1-beta)/(n_samp-1));
%             K(idx_J, idx_I) = K(idx_J, idx_I) * ((1-beta)/(n_samp-1));
            K(idx_I, idx_J) = K(idx_I, idx_J);
            K(idx_J, idx_I) = K(idx_J, idx_I);
        end
        
    end
end

disp 'computing operator'

switch kernel_symm
    case '+'
        K = K + K';
    case '*'
        K = K * K';
    case '.*'
        K = K .* K';
    otherwise
        disp('Unknown kernel symmetrization')
end

DiffOp = bsxfun(@rdivide, K, sum(K,2));

disp 'done'
