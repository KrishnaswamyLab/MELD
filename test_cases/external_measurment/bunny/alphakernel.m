function [axy,gxy] = alphakernel(data, varargin)
    p = inputParser;
    
    checkeps = @(x) isa(x, 'function_handle');
    addRequired(p,'data',@isnumeric);
    addParameter(p,'a',2,@isnumeric);
    addParameter(p,'eps','knn', checkeps);
    addParameter(p, 'k', 5, @isint);
    addParameter(p, 'npca', 5, @isnumeric);
    
    parse(p,data,varargin{:})
    
    data = p.Results.data;
    a = p.Results.a;
    epsfunc = p.Results.eps;
    k = p.Results.k;
    npca = p.Results.npca;
    
    
    M = svdpca(data, npca, 'random');
    disp('building kernel..');
    PDX = squareform(pdist(M));
    
    if ischar(epsfunc)
        knnDST = sort(PDX,1);
        eps = knnDST(k+1,:)';
    else
        eps = epsfunc(PDX);
    end
    
    PDX = bsxfun(@rdivide, PDX, eps);
    kxy = exp(-PDX.^a);
    axy = kxy+kxy';
    %kxy = kxy - diag(diag(kxy));
    gxy = axy/2;
    gxy = gxy - diag(diag(gxy));
end
    
function answer = isint(n)

    if size(n) == [1 1]
        answer = isreal(n) && isnumeric(n) && round(n) == n &&  n >0;
    else
        answer = false;
    end
end