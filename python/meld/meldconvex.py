import numpy as np
import pygsp
import graphtools
import graphtools.base
import scipy.sparse as sparse
import inspect
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from . import utils
from sklearn import preprocessing
from scipy.linalg import expm
from scipy.linalg import fractional_matrix_power as fmp
import sklearn.preprocessing as sklp
import warnings
from functools import partial


def _check_pygsp_graph(G):
    if isinstance(G, graphtools.base.BaseGraph):
        if not isinstance(G, pygsp.graphs.Graph):
            raise TypeError(
                "Input graph should be of type pygsp.graphs.Graph. "
                "With graphtools, use the `use_pygsp=True` flag.")
    else:
        raise TypeError(
            "Input graph should be of graphtools.base.BaseGraph."
            "With graphtools, use the `use_pygsp=True` flag.")


def meld(X, G, beta, offset=0, order=1, solver='chebyshev', M=50,
         lap_type='combinatorial', fi='regularizedlaplacian', alpha=2):
    """
    Performs convex meld on the input signal.
    This function solves:

        (1) :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \beta \| \nabla x\|_2^2`

        OR
        regularizes (1) using \inner{X, randomwalk(L)*X}, the p-step random walk (math to come)

    Note the nice following relationship for (1):

        (2) :math:`x^T L x = \| \nabla x\|_2^2`

    Also note that the solution to (1) may be phrased as the lowpass filter:

        (3) :math:`sol = h(L)x` with :math:`h(\lambda) := \frac{1}{1+\beta\lambda}`

    We use (3) by default as it is faster in the case of few input signals.


    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input signal array to meld.
    G : graphtools.Graph object
        Graph to perform data smoothing over.
    beta : int
        Amount of smoothing to apply.
        Acts as 'p' parameter if fi == 'randomwalk'
    offset: int, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.
        Recommend using an eigenvalue from g based on the
        spectral distribution.
    order: int, optional, Default: 1
        Falloff and smoothness of the filter.
        High order leads to square-like filters.
    Solver : string, optional, Default: 'chebyshev'
        Method to solve convex problem.
        'cheby' uses a chebyshev polynomial approximation of the corresponding filter
        'exact' uses the eigenvalue solution to the problem
        'matrix' is deprecated and may not function appropriately
    M : int, optional, Default: 50
        Order of chebyshev approximation to use.
    fi: string, optional, Default: 'regularizedlaplacian'
        Filter to use for (1).
        'regularizedlaplacian' is the exact solution of (1)
        'randomwalk' is a randomwalk polynomial that is related to diffusion via rw = ((alpha-1)I+P)^t

    Returns
    -------
    sol : ndarray [n, p]
        2 dimensional array of smoothed input signals
    """

    if not isinstance(solver, str):
        raise TypeError("Input method should be a string")
    solver = solver.lower()

    if not isinstance(fi, str):
        raise TypeError("Input filter should be a string")
    fi = fi.lower()
    if fi not in ['regularizedlaplacian', 'randomwalk']:
        raise NotImplementedError(
            '{} filter is not currently implemented.'.format(fi))

    _check_pygsp_graph(G)

    if not isinstance(lap_type, str):
        raise TypeError("Input lap_type should be a string")
    lap_type = lap_type.lower()
    Gbak = G
    if G.lap_type != lap_type:
        warnings.warn(
            "Changing lap_type may require recomputing the Laplacian")
        G.compute_laplacian(lap_type)

    if X.shape[0] != G.N:
        if len(X.shape) > 1 and X.shape[1] == G.N:
            print(
                "input matrix is column-wise rather than row-wise. "
                "transposing (output will be transposed)")
            X = X.T
        else:
            raise ValueError(
                "Input data ({}) and input graph ({}) "
                "are not of the same size".format(X.shape, G.N))

    if fi == 'randomwalk':
        # used for random walk stochasticity
        D = sparse.diags(np.ravel(np.power(G.W.sum(0), -1)), 0).tocsc()
    if solver == 'matrix':
        # use matrix inversion / powering
        I = sparse.identity(G.N)
        if fi == 'regularizedlaplacian':  # fTLf
            mat = sparse.linalg.inv(
                (I + np.matrix_power(beta * G.L - offset * I, order)).tocsc())
        elif fi == 'randomwalk':  # p-step random walk
            mat = (alpha * I - (G.L * D))**beta

        sol = mat.T @ X  # apply the matrix
        sol = np.squeeze(np.asarray(sol))  # deliver a vector

    else:
        # use approximations
        if fi == 'regularizedlaplacian':  # fTLf
            def filterfunc(x): return 1 / (1 + (beta * x - offset)**order)

        elif fi == 'randomwalk':  # p-step random walk
            L_bak = G.L
            # change the eigenbasis by normalizing by degree (stochasticity)
            G.L = (L_bak * D).T

            def filterfunc(x): return (alpha - x)**beta

        G.estimate_lmax()
        filt = pygsp.filters.Filter(G, filterfunc)  # build filter
        sol = filt.filter(X, method=solver, order=M)  # apply filter
        if fi == 'randomwalk':
            G.L = L_bak  # restore L

    sol = utils.convert_to_same_format(sol, X)

    Gout = G
    G = Gbak

    return sol


class MELDCluster(BaseEstimator):

    def __init__(self, n_clusters=10, window_count=9, window_sizes=None, window=None,
                 spectral_init=False, initial_labels=None, suppress=False, **kwargs):
        """MELDCluster

        Parameters
        ----------
        n_clusters : int, optional
            Description
        window_count : int, optional
            Number of windows to use if window_sizes = None
        window_sizes : None, optional
            ndarray of integer window sizes to supply to t
        window : None, optional
            Window matrix.  Not supported
        suppress : bool, optional
            Suppress warnings
        **kwargs
            Description

        Raises
        ------
        NotImplementedError
            Window functions are not implemented

        Examples
        --------

        """
        self.suppress = suppress
        if window is not None:
            raise NotImplementedError(
                "User defined windows have not been implemented.")
        else:
            self._basewindow = None
        if window_sizes is None:
            self.window_sizes = np.power(2, np.arange(window_count))
        else:
            self.window_sizes = window_sizes

        self.window_count = np.min(self.window_sizes.shape)
        self._n_clusters = n_clusters
        self._spectral_init = spectral_init
        self._initial_labels = None
        self._initial_centroids = None
        self._clusterobj = KMeans(n_clusters=n_clusters, **kwargs)

        self.initial_labels = initial_labels
        self._h = None
        self._U = None
        self._N = None
        self._Cs = None
        self._C = None
        self._normL = None
        self._SCbasis = None
        self._isfit = False
        self.__sklearn_params = kwargs
        # print(**kwargs)

    def _activate(self, x, alpha=1):
        """_activate: activate spectrograms for clustering

        Parameters
        ----------
        x : numeric
            input signal
        alpha : int, optional
            amount of activation

        Returns
        -------
        activated signal
        """
        return np.tanh(alpha * np.abs(x))

    def _compute_spectrogram(self, s, window):
        """_compute_spectrogram: computes spectrograms for
        arbitrary window/signal/graph combinations

        Parameters
        ----------
        s : no.ndarray
            Input signal
        U : np.ndarray
            eigenvectors
        window : TYPE
            window matrix

        Returns
        -------
        C
            Normalized Spectrogram

        Raises
        ------
        TypeError
            Description
        """
        if self._U is None:
            raise ValueError(
                'Estimator must be `fit` before running `_compute_spectrogram`.')
        if sparse.issparse(window):
            warnings.warn("sparse windows not supported."
                          "Casting to np.ndarray.")
            window = window.toarray()

        else:
            if not isinstance(window, np.ndarray):
                raise TypeError(
                    "window must be a numpy.array or"
                    "scipy.sparse.csr_matrix.")
        C = np.multiply(window, s[:, None])
        C = sklp.normalize(self._U.T@C, axis=0)
        return C.T

    def _compute_window(self, window, **kwargs):
        """_compute_window
        apply operation to window function

        Parameters
        ----------
        window : TYPE
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        TypeError
            Description
        """
        if sparse.issparse(window):
            warnings.warn("sparse windows not supported."
                          "Casting to np.ndarray.")
            window = window.toarray()

        else:
            if not isinstance(window, np.ndarray):
                raise TypeError(
                    "window must be a numpy.array or"
                    "scipy.sparse.csr_matrix.")
            else:
                window = np.linalg.matrix_power(window, kwargs['t'])
                return sklp.normalize(window, 'l2', axis=0).T

    def fit(self, G, refit=False, **kwargs):
        _check_pygsp_graph(G)
        if self._isfit and not refit:
            warnings.warn("Estimator is already fit. "
                          "Call MELDCluster.fit(G,refit=True)"
                          " to refit")
            return self
        if self._basewindow is None:
            if sparse.issparse(G.diff_op):
                if not self.suppress:
                    warnings.warn("sparse windows not supported."
                                  "Casting to np.ndarray.")
                self._basewindow = G.diff_op.toarray()
            else:
                self._basewindow = G.diff_op
        self._h = np.zeros((G.N, G.N, self.window_count)
                           )

        for i, t in enumerate(self.window_sizes):
            self._h[:, :, i] = self._compute_window(
                self._basewindow, t=t).astype(float)
        # check the gft basis..
        if G.lap_type == 'normalized':
            warnings.warn('Combinatorial Laplacian is required for'
                          ' spectrogram clustering.  Computing eigenvectors'
                          ' of the combinatorial Laplacian.')
            if sparse.issparse(G.W):
                W = G.W.toarray()
            else:
                W = G.W
            L = np.diag(G.dw) - W
            _, self._U = np.linalg.eigh(L)
        else:
            self._U = G.U
        self._N = G.N

        self.__compute_spectral_clusters(G, recompute=True)
        self._isfit = True
        return self

    def transform(self, s, center=True, **kwargs):
        if not self.suppress:
            if (self._C is not None or self._Cs is not None):
                warnings.warn("Overwriting previous spectrogram. "
                              "Suppress this warning with "
                              "MELDCluster(suppress=True)")
        if not self._isfit:
            if not self.suppress:
                warnings.warn("Estimator is not fit. "
                              "Call MELDCluster.fit(). ")
            return None
        else:
            if not isinstance(s, (list, tuple, np.ndarray)):
                raise TypeError('Input signal s must be an array')
            s = np.array(s)
            if self._N not in s.shape:
                raise ValueError('At least one axis of s must be'
                                 ' of length N.')
            else:
                if center:
                    s = s - s.mean()
                self._C = np.zeros((self._N, self._N))
                self._Cs = np.zeros((
                    self._N, self._N, self.window_count))
                for t in range(self.window_count):
                    temp = self._compute_spectrogram(
                        s, self._h[:, :, t])
                    self._Cs[:, :, t] = temp
                    temp = self._activate(temp)
                    temp = sklp.normalize(temp, 'l2', axis=1)

                    self._C += temp
                """ This can be added later to support multiple signals
                for i in range(ncols):
                    for t in range(self.window_count):
                        temp = self._compute_spectrogram(
                            s[:, i], self._U, self._h[:, :, t])
                        if self._activated:
                            temp = self._activate(
                                temp)
                        if self._store_Cs:
                            self._Cs[:, :, t, i] = temp
                        else:
                            self._C[:, :, i] += temp"""

        return self._C

    def fit_transform(self, G, s, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(s, **kwargs)

    def predict(self, s=None, **kwargs):
        if not self._isfit:
            warnings.warn("Estimator is not fit. "
                          "Call MELDCluster.fit(). ")
            return None
        if self._C is None and s is None:
            warnings.warn("Estimator has no spectrogram to cluster. "
                          "Call MELDCluster.transform(s). ")
            return None
        elif s is not None and self._C is not None:
            self.transform(s, **kwargs)

        # make sure we are on the correct set of centroids.
        self.set_kmeans_params(init=self.initial_centroids)
        self.labels_ = self._clusterobj.fit_predict(self._C)
        return self.labels_

    def fit_predict(self, G, s, **kwargs):
        self.fit_transform(G, s, **kwargs)
        return self.predict()

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, newk):
        self._n_clusters = newk
        self._clusterobj.set_params(n_clusters=self._n_clusters)
        self.__check_matching_k_initialization()

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop('k', False)
        if k:
            self._n_clusters = k
        self._sklearn_params = kwargs
        self._clusterobj.set_params(n_clusters=self._n_clusters, **kwargs)
        self.__check_matching_k_initialization()

    @property
    def initial_labels(self):
        return self._initial_labels

    @initial_labels.setter
    def initial_labels(self, initial_labels):
        if not isinstance(initial_labels,
                          (type(None), list, tuple, np.ndarray)):
            raise TypeError('Initial labels must be an array')
        elif initial_labels is not None:
            initial_labels = np.array(initial_labels)
        self._initial_labels = initial_labels

    @property
    def initial_centroids(self):
        if self._C is not None:
            if (isinstance(self._initial_centroids, np.ndarray) or
                    self._initial_centroids in (None, 'k-means++', 'random')):
                if self.initial_labels is not None:
                    self._initial_centroids = self.__compute_centroids(
                        self._initial_labels)
                else:
                    self._initial_centroids = None
        if self._initial_centroids is None:
            self._initial_centroids = 'k-means++'
        return self._initial_centroids

    def __compute_centroids(self, labels):
        ncols = len(np.unique(labels))
        B = np.zeros([self._N, ncols])
        B[(np.arange(self._N), labels.squeeze())] = 1
        out = np.zeros((ncols, self._N))
        for i in range(ncols):
            out[i, :] = self._C[B.astype(bool)[:, i], :].mean(0)
        return out

    def __compute_spectral_clusters(self, G=None, recompute=False):
        if self._spectral_init:

            if self._initial_labels is not None and not recompute:
                return self

            elif self._initial_labels is not None and recompute:
                warnings.warn("Overwriting current initial"
                              " labels with spectral clusters")

            if self._SCbasis is None or recompute:
                self._SCbasis = None
                if G is None:
                    raise RuntimeError('To compute spectral clusters'
                                       'a graph must be supplied.')

                if sparse.issparse(G.L):
                    L = G.L.toarray()
                else:
                    L = G.L

                if G.lap_type == 'normalized':
                    self._normL = L
                    if hasattr(G, '_U'):  # precomputed eigs, easy
                        self._SCbasis = G._U[:, :self._n_clusters]
                else:
                    # combinatorial laplacian, we need the normalized one
                    sqrdeg = np.sqrt(G.dw)
                    if any(sqrdeg == 0):
                        raise ValueError("Graph has isolated vertices")
                    else:
                        sqrdeg = np.diag(sqrdeg ** -1)
                        self._normL = sqrdeg@L@sqrdeg

                if self._SCbasis is None:
                    _, self._SCbasis = sparse.linalg.eigsh(
                        self._normL, self._n_clusters, which='SM')

                self._SCbasis = sklp.normalize(self._SCbasis, axis=1)
            self.set_kmeans_params(init='k-means++')
            self.initial_labels = self._clusterobj.fit_predict(self._SCbasis)
        return self

    def __check_matching_k_initialization(self):
        if self._initial_labels is not None:
            if self.initial_centroids.shape[0] != self._n_clusters:
                if self._spectral_init and self._isfit:
                    # this implies that we have already run spectral clustering
                    # we need to recompute it
                    warnings.warn('New _n_clusters does not match '
                                  ' spectral initialization. '
                                  'Reclustering spectral labels. ')
                    self.initial_labels = self._clusterobj.fit_predict(
                        self._SCbasis)
                    self.set_kmeans_params(init=self.initial_centroids)
                elif not self._spectral_init:
                    warnings.warn('New _n_clusters does not match initial labels '
                                  'Discarding initial labels.')
                    self.initial_labels = None
                    self.set_kmeans_params(init=self.initial_centroids)
