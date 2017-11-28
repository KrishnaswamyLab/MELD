function [B, W1] = random_projection(A, K)
% [B, W1] = random_projection(A, K)
%   A is data
%   K is rank of gaussian random matrix (default 100)
%   B is random projection
%   W1 is
%
%   A = B * W1'
%   A_new = B_new * W1'
%
%   to do PCA on projection:
%   X = bsxfun(@minus, B, mean(B));
%   [U,~,~] = svd(X','econ');
%   pc = X * U;

if ~exist('K', 'var')
    K = 100;
end

N = size(A,1);
P = min(K,N);
X = randn(N,P);
Y = A' * X;
W1 = orth(Y);
B = A * W1;