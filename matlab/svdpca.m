function [Y, U, mu, S] = svdpca(X, k, method)

if ~exist('method','var')
    method = 'svd';
end

mu = mean(X);
X = bsxfun(@minus, X, mu);

switch method
    case 'svd'
        disp 'PCA using SVD'
        [U,S,~] = svds(X', k);
        Y = X * U;
    case 'random'
        disp 'PCA using random SVD'
        [U,S,~] = randPCA(X', k);
        Y = X * U;
        S = diag(S);
    case 'none'
        disp 'No PCA performed'
        Y = X;
end
