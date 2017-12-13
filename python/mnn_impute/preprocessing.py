# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import sklearn.preprocessing
import sklearn.decomposition
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA

# Classical MDS
def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)
    Copyright © 2014-7 Francis Song, New York University
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/float(n)

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals

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
    X = X - np.mean(X) # mean centering

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

def center_batches(data, sample_idx=None, verbose=False):
    """Center the data by batch
    Assumes input has n rows by m features
    The n rows are split into k batches, indexed by sample_idx
    We force the mean of each feature to be 0 across each batch

    Parameters
    ----------
    data : ndarray [n,p]
        2 dimensional input data array with n cells and p dimensions

    Returns
    -------
    data_norm : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    if sample_idx is None: 
        if verbose: print("sample_idx not given, centering data as a single batch")
        sample_idx = np.repeat([0], data.shape[0])
    if verbose: print("Centering data by batch for %s cells and %s batches"%(data.shape[0], len(np.unique(sample_idx))))
    data_norm = np.zeros(shape=data.shape)
    for sample in np.unique(sample_idx):
        data_norm[sample_idx == sample] = sklearn.preprocessing.scale(data[sample_idx == sample], axis=0, with_mean=True, with_std=False)
    return data_norm
