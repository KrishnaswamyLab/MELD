function prcts = batch_quantify(x, labels, k,npca, distfun)
    %batch_quantify: measure nearest neighbor confusion matrix in data x.
    % compares true labels to neighbor distribution, averaged over all
    % samples in label
%   [Y, K, L] = hpf0(x,labels, k, npca, distfun)
%      Inputs:
%           x: input data to be measured
%           labels: integer labels starting at 0, incrementing by 1. eg 
%                   [0 1 2 3] NOT [0 2 3 6]
%           k: nearest neighbor parameter for #neighbors to survey
%           npca: number of principal components to compute distances over
%           distfun: distance function pass to pdist
    

    % parse parameters
    M = svdpca(x, npca, 'random');

    rnge = unique(labels); % range of labels
    
    dists = squareform(pdist(M,distfun)); %compute distances
    sz = size(dists, 1);
    [~,ix] = sort(dists);
    nnlabels = labels(ix(2:k+1,:)); %find k-nearest neighbors
    
    comp = bsxfun(@eq, nnlabels(:),rnge); %compare knn labels to labels for logical 
    comp = reshape(comp, sz, k, size(rnge,2)); % reshape to 3D, each dim is a label
    
    comp2 = (sum(comp,2) ./ k); %percent of labels in each point
    comp2 = squeeze(comp2); %house keeping
    
    trueval = labels(ix(1,:)); %pt label
    trueval = bsxfun(@eq, trueval(:), rnge); % match to range - may not be necessary?
    prcts = (comp2' * trueval)./sum(trueval); %"Confusion Matrix"
end