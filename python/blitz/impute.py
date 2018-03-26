import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.metrics import mutual_info_score


def mnn_kernel(X, k, a, beta=0, sample_idx=None, kernel_symm='+', metric='euclidean', verbose=False):
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

    beta: float (0:1]
        This parameter weights the MNN kernel. Values closer to 1 increase batch
        correction.

    sample_idx : ndarray [n], optional, default: None
        1 dimensional array specifying the sample to which each observation in
        X belongs. If left empty, X is assumed to be one sample

    kernel_symm : string, optional, default: '+'
        Defines method of MNN symmetrization.
        '+'  : additive
        '*'  : multiplicative
        '.*' : inner product

    metric : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        Specifies distance metric for finding MNN

    Returns
    -------
    diff_op : ndarray [n, n]
        2 dimensional array diffusion operator created using a MNN kernel
    """
    one_sample = (sample_idx is None) or (np.sum(sample_idx) == len(sample_idx))
    if not one_sample:
        if not (0 < beta <= 1):
            raise ValueError('Beta must be in the half-open interval (0:1]')
    else:
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
            if si == sj:
                if one_sample:
                    K.iloc[sample_idx == si, sample_idx == sj] = k_ij  # fill out values in K for NN on diagnoal
                else:
                    K.iloc[sample_idx == si, sample_idx == sj] = k_ij * (1 - beta) # fill out values in K for NN on diagnoal
            else:
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij  # fill out values in K for NN on diagnoal
                # now go back and do J -> I
                pdx_ji = pdx_ij.T # Repeat to find KNN from J -> I
                kdx_ji = np.sort(pdx_ji, axis=1)
                e_ji   = kdx_ji[:,k]
                pdxe_ji = pdx_ji / e_ji[:, np.newaxis]
                k_ji = np.exp(-1 * (pdxe_ji** a))
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij  # fill out values in K for NN on diagnoal

    if verbose: print('Computing Operator...')
    if kernel_symm=='+':
        K = K + K.T
    elif kernel_symm=='*':
        K = K @ K.T
    elif kernel_symm=='.*':
        K = K * K.T


    K = np.multiply(K, K.T)
    diff_deg = np.diag(np.sum(K,0)) # degrees
    diff_op = np.dot(np.diag(np.diag(diff_deg)**(-1)),K)
    if verbose: print('Done!')
    return diff_op

def magic(X, diff_op, t='auto', verbose=False):
    if isinstance(t, int):
        if verbose: print('powering operator')
        diff_op_t = np.linalg.matrix_power(diff_op, t)
        return np.dot(diff_op_t, X)

    elif t == 'auto':
        data_imputed = X



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

def calc_MI(x, y, bins=8):
    """
    Calculates mutual information between X and Y.

    Parameters
    ----------
    X : array [p]
        Values for X

    Y : array [p]
        Values for Y

    bins : int
        Number of histogram bins for calculating mutual information.

    Returns
    -------
    mi : array [p]
        Array with mutual information scores
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
