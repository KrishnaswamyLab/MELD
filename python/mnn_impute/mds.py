# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np

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

def embed_MDS(X, ndim=2, how='classic', distance_metric='euclidean', njobs=1, seed=None):
    """
    Performs classic, metric, and non-metric MDS

    Parameters
    ----------
    X: ndarray [n_samples, n_samples]
        2 dimensional input data array with n_samples
        embed_MDS does not check for matrix squareness, but this is nescessary for PHATE

    n_dim : int, optional, default: 2
        number of dimensions in which the data will be embedded

    how : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    distance_metric : string, optional, default: 'euclidean'
        choose from [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        distance metric for MDS

    njobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used

    seed: integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Returns
    -------
    Y : ndarray [n_samples, n_dim]
        low dimensional embedding of X using MDS
    """

    ## MDS embeddings, each gives a different output.
    X_dist = squareform(pdist(X, distance_metric))

    if how == 'classic':
        #classical MDS as defined in cmdscale
        Y = cmdscale(X_dist)[0][:,:ndim]
    elif how == 'metric':
        #Metric MDS from sklearn
        Y = MDS(n_components=ndim, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=njobs,
                     n_init=1).fit_transform(X_dist)
    elif how == 'nonmetric':
        Y_mmds = MDS(n_components=ndim, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=njobs,
                     n_init=1).fit_transform(X_dist)
        #Nonmetric MDS from sklearn using metric MDS as an initialization
        Y = MDS(n_components=ndim, metric=False, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=njobs,
                     n_init=1).fit_transform(X_dist,init=Y_mmds)
    else:
        raise ValueError("Allowable 'how' values for MDS: 'classic', 'metric', or 'nonmetric'. '%s' was passed."%(how))
    return Y
