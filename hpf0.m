function [Y,K,L] = hpf0(x, varargin)
    %hpf0  HIGH PASS FILTER 0: Remove first fourier coefficient from data 
    % using partial eigendecomposition
%   [Y, K, L] = hpf0(x,varargin)
%      Inputs:
%           x: input data to be filtered
%      varargin:
%           a: Alpha Decay of kernel, 
%               default: 2
%           eps: lambda function handle for kernel bandwidth, 
%               default: string('knn') -> adaptive bandwidth
%           k: knn parameter for adaptive bandwidth, 
%               default: 5
%           npca: number of pca components to use when building kernel,
%               default: 100

    

    % parse parameters
    p = inputParser;
    checkeps = @(x) isa(x, 'function_handle');
    addRequired(p,'x',@isnumeric);
    addParameter(p,'a',2,@isnumeric);
    addParameter(p,'eps',"knn", checkeps); %default is knn, but lambda functions work if defined on the distance matrix
    addParameter(p, 'k', 5, @isint);
    addParameter(p, 'npca', 100, @isnumeric);
    
    parse(p,x,varargin{:})
    
    x = p.Results.x;
    a = p.Results.a;
    epsfunc = p.Results.eps;
    k = p.Results.k;
    npca = p.Results.npca;
    
    % math
    K = alphakernel(x, a, epsfunc, k, npca); % build graph kernel (not affinity)
    D = sum(K, 1); % degrees
    D = diag(D); % diagonal degrees
    L = D-K; % graph laplacian
    [v1,~] = eigs(L, 1, 'smallestabs'); %get first eigenvector
    
    a0 = v1'*x; %fourier coefficient matrix (frequency domain)
    f0 = a0.*v1; %component of function that is in a0 (vertex domain)
    
    Y = x - f0; % from the definition of the inverse fourier transform
    
    
    
end

function gxy = alphakernel(x,a,epsfunc,k,npca)
    
    M = svdpca(x, npca, 'random');
    disp("building kernel..");
    PDX = squareform(pdist(M));
    
    if isstring(epsfunc) % any string defaults to knn
        knnDST = sort(PDX,1);
        eps = knnDST(k+1,:)';
    else
        eps = epsfunc(PDX);
    end
    
    PDX = bsxfun(@rdivide, PDX, eps);
    kxy = exp(-PDX.^a);
    gxy = kxy+kxy';
    gxy = gxy/2;
    gxy = gxy - diag(diag(gxy));
end
    
function answer = isint(n)

    if size(n) == [1 1]
        answer = isreal(n) && isnumeric(n) && round(n) == n &&  n >0;
    else
        answer = false;
    end
end