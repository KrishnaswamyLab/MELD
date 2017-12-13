# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import sklearn.preprocessing
import sklearn.decomposition
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA

def svdpca(X, n_components, method='svd', verbose=False):
    """
    Mean-center and applies a number of SVD and PCA functions to data matrix X
    and returns the transformed matrix.

    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input data array with n observations and p dimensions

    n_components : int
        number of dimensions in which the data will be embedded

    method : string, optional, default: 'svd'
        options: ['svd', 'random', None]
        The method of dimensionality reduction to be performed

    Returns
    -------
    Y : ndarray [n, n_components]
        2 dimensional array transformed using specified dimensionality reduction
        method
    """
    X = X - X.mean() # mean centering

    if method == 'svd':
        if verbose: print('PCA using SVD')
        U, S, V = svds(X.T, k=n_components)
        Y = np.dot(X, U)
    elif method == 'random':
        if verbose: print('PCA using random SVD')
        Y = PCA(n_components=n_components, svd_solver='randomized').fit_transform(X)
    elif method == None:
        if verbose: print('No PCA performed')
        Y = X;
    else:
        raise NotImplementedError('PCA method %s has not been implemented.'%method)

    return Y

def library_size_normalize(data, verbose=False):
    """Performs L1 normalization on input data
    Performs L1 normalization on input data such that the sum of expression values for each cell sums to 1
    then returns normalized matrix to the metric space using median UMI count per
    cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : ndarray [n,p]
        2 dimensional input data array with n cells and p dimensions

    Returns
    -------
    data_norm : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    if verbose: print("Normalizing library sizes for %s cells"%(data.shape[0]))
    data_norm = sklearn.preprocessing.normalize(data, norm = 'l1', axis = 1)
    #norm = 'l1' computes the L1 norm which computes the
    #axis = 1 independently normalizes each sample

    median_transcript_count = np.median(data.sum(axis=1))
    data_norm = data_norm * median_transcript_count
    return data_norm
