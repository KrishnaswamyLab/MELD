function prcts = batch_quantify(x, labels, k,npca, distfun)
    %batch_quantify: measure nearest neighbor confusion matrix in data x.
    % compares true labels to neighbor distribution, averaged over all
    % samples in label. OUTPUT IS asymmetric because we are looking from
    % j to all neighbors k, which is not symmetric.  Row-wise
    % interpretation is best. 
%   prcts = batch_quantify(x,labels, k, npca, distfun)
%      Inputs:
%           x: input data to be measured
%           labels: integer labels starting at 1, incrementing by 1. eg 
%                   [1 1 2 3 3 4 1 5 ] NOT [0 2 2 3 6] <- no 1, 4 or 5 labels
%           k: nearest neighbor parameter for #neighbors to survey
%           npca: number of principal components to compute distances over
%           distfun: distance function pass to pdist
    

    % parse parameters
    M = svdpca(x, npca, 'random');
    if min(labels) == 0
        labels = labels +1;
    end
    rnge = unique(labels); % range of labels
    dists = squareform(pdist(M,distfun)); %compute distances
    sz = size(dists, 1);
    [~,ix] = sort(dists);
    nnlabels = labels(ix(1:k+1,:)); %find k-nearest neighbors
    curLs = nnlabels(1,:);
    nnlabels = sort(nnlabels(2:end,:));
    prcts = zeros(max(labels),max(labels));
    nClass = zeros(max(labels),1);

    for i=1:sz
        curL = curLs(i);
        [C,ia] = unique(nnlabels(:,i));
        prctvec = zeros(max(labels), 1)';
        for j = 1:numel(ia)
            if j == numel(ia)
                slice = nnlabels(ia(j):end,i);
            else
                slice = nnlabels(ia(j):ia(j+1)-1,i);
                
        end
            prctvec(C(j)) = size(slice,1);
        end
        prctvec = prctvec/k;

        prcts(curL,:) = prcts(curL,:) + prctvec;
        nClass(curL) = nClass(curL) + 1;
    end     
    
    prcts = prcts ./ nClass;
end
