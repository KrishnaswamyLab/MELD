function [DiffOp, K] = mnn_kernel(data, sample_ind, npca, k, a, distfun)
% [DiffOp, K] = mnn_kernel(data, sample_ind, npca, k, a, distfun)
%
%   creates a kernel that in combination with MAGIC does batch correction
%
%   sample_ind are the sample indices

if ~exist('distfun','var')
    distfun = 'euclidean';
end

if isempty(sample_ind)
    sample_ind = ones(size(data,1),1);
end

uniq_samp = unique(sample_ind);
n_samp = length(uniq_samp);

if npca > 0
    M = svdpca(data, npca, 'random');
else
    M = data;
end

K = nan(size(data,1));
for I=1:n_samp
    I
    samp_I = uniq_samp(I);
    idx_I = sample_ind == samp_I;
    MI = M(idx_I,:);
    for J=1:I
        samp_J = uniq_samp(J);
        idx_J = sample_ind == samp_J;
        MJ = M(idx_J,:);
        PDXIJ = pdist2(MI, MJ, distfun);
        knnDSTIJ = sort(PDXIJ,2);
        epsilonIJ = knnDSTIJ(:,k);
        PDXIJ = bsxfun(@rdivide,PDXIJ,epsilonIJ);
        KIJ = exp(-PDXIJ.^a);
        K(idx_I, idx_J) = KIJ;
        if I~=J
            PDXJI = PDXIJ';
            knnDSTJI = sort(PDXJI,2);
            epsilonJI = knnDSTJI(:,k);
            PDXJI = bsxfun(@rdivide,PDXJI,epsilonJI);
            KJI = exp(-PDXJI.^a);
            K(idx_J, idx_I) = KJI;
        end
    end
end

disp 'computing operator'

K = K + K'; % MNN
DiffDeg = diag(sum(K,2)); % degrees
DiffOp = DiffDeg^(-1)*K; % row stochastic

disp 'done'

