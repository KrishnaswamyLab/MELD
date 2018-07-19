function K = kernel_sparse(MI, MJ, k, distfun)
% K = kernel_sparse(M1, M2, k)

N = size(MI, 1);
M = size(MJ, 1);

idx = knnsearch(MJ, MI, 'k', k, 'dist', distfun);

i = repmat((1:N)',1,size(idx,2));
i = i(:);
j = idx(:);
K = sparse(i, j, ones(size(j)), N, M);
