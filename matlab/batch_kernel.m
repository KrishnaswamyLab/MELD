function dataA_new = batch_kernel(dataA, dataB, npca, k, a, t, distfun)

if npca > 0
    disp 'doing PCA'
    MA = svdpca(dataA, npca, 'random');
    MB = svdpca(dataB, npca, 'random');
else
    MA = dataA;
    MB = dataB;
end

disp 'computing distances'
PDX = pdist2(MA, MB, distfun);
knnDST = sort(PDX,2);
disp 'computing kernel'
epsilon = knnDST(:,k);
PDX = bsxfun(@rdivide,PDX,epsilon);
K = exp(-PDX.^a);
K = K * K';
disp 'computing operator'
DiffOp = bsxfun(@rdivide, K, sum(K,2));
disp 'powering operator'
DiffOp_t = DiffOp^t;
disp 'batch correcting'
dataA_new = (eye(size(DiffOp_t)) - DiffOp_t) * dataA;
disp 'done'
