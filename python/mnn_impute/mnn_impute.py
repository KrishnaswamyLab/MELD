import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, squareform, pdist

def mnn_kernel(X, k, a, sample_idx=None, metric='euclidean', verbose=False):
    """
    Creates a kernel linking the k mutual nearest neighbors (MNN) across datasets
    and performs diffusion on this kernel using MAGIC to apply batch correction.

    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input data array with n observations and p dimensions

    k : int
        Number of neighbors to use

    a : int
        Specifies alpha for the α-decaying kernel

    sample_idx : ndarray [n], optional, default: None
        1 dimensional array specifying the sample to which each observation in
        X belongs. If left empty, X is assumed to be one sample

    metric : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        Specifies distance metric for finding MNN

    Returns
    -------
    diff_op : ndarray [n, n]
        2 dimensional array diffusion operator created using a MNN kernel
    """

    if sample_idx is None:
        sample_idx = np.ones(len(X))

    samples = np.unique(sample_idx)

    K = np.zeros((len(X), len(X)))
    K[:] = np.nan
    K = pd.DataFrame(K)

    # Build KNN kernel
    if verbose: print('Finding KNN...')
    for si in samples:
        X_i = X[sample_idx == si]            # get observations in sample i
        for sj in samples:
            X_j = X[sample_idx == sj]        # get observation in sample j
            pdx_ij = cdist(X_i, X_j, metric=metric) # pairwise distances
            kdx_ij = np.sort(pdx_ij, axis=1) # get kNN
            e_ij   = kdx_ij[:,k]             # dist to kNN
            pdxe_ij = pdx_ij / e_ij[:, np.newaxis] # normalize
            k_ij   = np.exp(-1 * (pdxe_ij ** a))  # apply α-decaying kernel
            K.iloc[sample_idx == si, sample_idx == sj] = k_ij # fill out values in K for NN from I -> J
            if si != sj:
                pdx_ji = pdx_ij.T # Repeat to find KNN from J -> I
                kdx_ji = np.sort(pdx_ji, axis=1)
                e_ji   = kdx_ji[:,k]
                pdxe_ji = pdx_ji / e_ji[:, np.newaxis]
                k_ji = np.exp(-1 * (pdxe_ji** a))
                K.iloc[sample_idx == sj, sample_idx == si] = k_ji
    if verbose: print('Computing Operator...')
    K = K + K.T
    diff_deg = np.diag(np.sum(K,0)) # degrees
    diff_op = np.dot(np.diag(np.diag(diff_deg)**(-1)),K)
    if verbose: print('Done!')
    return diff_op

def magic(X, diff_op, t, verbose=False):
    if verbose: print('powering operator')
    diff_op_t = np.linalg.matrix_power(diff_op, t)
    return np.dot(diff_op_t, X)

# computes kernel and operator
def get_operator(data=None, k=5, a=10):
    pdx = squareform(pdist(data, metric='euclidean')) # compute distances on pca
    knn_dst = np.sort(pdx, axis=1) # get knn distances for adaptive kernel
    epsilon = knn_dst[:,k] # bandwidth(x) = distance to k-th neighbor of x
    pdx = (pdx / epsilon).T # autotuning d(x,:) using epsilon(x).
    gs_ker = np.exp(-1 * ( pdx ** a)) # alpha decaying Gaussian kernel: exp(-D^alpha)
    gs_ker = gs_ker + gs_ker.T # symmetrization
    diff_deg = np.diag(np.sum(gs_ker,0)) # degrees
    diff_op = np.dot(np.diag(np.diag(diff_deg)**(-1)),gs_ker) # row stochastic -> Markov operator
    return diff_op

def normalize_imputed_vector(v, sample_idx):
    """
    v is an imputed discrete vector and sample_idx identifies
    which values of v belong to condition 0 (neg) or 1 (pos)
    """
    v_norm = (v - np.min(v)) / np.max(v - np.min(v))    # rescale v to [0,1]
    v_norm = v_norm / np.sum(sample_idx == 0)       # adjust by number of cells in sample 0
    v_norm_inv = (1 - v_norm) / np.sum(sample_idx == 1) # adjust by number of cells in sample 1
    v_norm_sum = v_norm_inv + v_norm
    v_norm = v_norm / v_norm_sum
    return v_norm
