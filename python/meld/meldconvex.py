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
                 suppress=False, **kwargs):
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
        self._h = None
        self._U = None
        self._N = None
        self._Cs = None
        self._C = None
        self._isfit = False
        self._input_signal = None
        self.__sklearn_params = kwargs
        print(**kwargs)
        self._clusterobj = KMeans(n_clusters=n_clusters, **kwargs)

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

    def _compute_spectrogram(self, input_signal, window):
        """_compute_spectrogram: computes spectrograms for
        arbitrary window/signal/graph combinations

        Parameters
        ----------
        input_signal : np.ndarray
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
            raise ValueError('Estimator must be `fit` before running `_compute_spectrogram`.')

        #TODO: Why is this sparse if it's no supported?
        if sparse.issparse(window):
            warnings.warn("sparse windows not supported."
                              "Casting to np.ndarray.")
            window = window.toarray()

        else:
            if not isinstance(window, np.ndarray):
                raise TypeError(
                    "window must be a numpy.array or"
                    "scipy.sparse.csr_matrix.")

        self._input_signal = input_signal
        C = np.multiply(window, self._input_signal[:, None])
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
        ## Would be nice if this didn't get called all the time
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

    def fit(self, G, refit=False):
        '''Sets eigenvectors and windows.'''

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
        self._U = G.U
        self._N = G.N
        self._isfit = True
        return self

    def transform(self, input_signal, center=True):
        self._input_signal = input_signal
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
            if not isinstance(input_signal, (list, tuple, np.ndarray)):
                raise TypeError('`input_signal` must be an array')
            input_signal = np.array(input_signal)
            if self._N not in input_signal.shape:
                raise ValueError('At least one axis of input_signal must be'
                                 ' of length N.')
            else:
                input_signal = input_signal - input_signal.mean()
                self._C = np.zeros((self._N, self._N))
                self._Cs = np.zeros((
                    self._N, self._N, self.window_count))
                for t in range(self.window_count):
                    temp = self._compute_spectrogram(
                        input_signal, self._h[:, :, t])
                    # There's maybe something wrong here
                    self._Cs[:, :, t] = temp
                    #temp = self._activate(temp)
                    #temp = sklp.normalize(temp, 'l2', axis=1) # This work goes nowhere

                self._C = np.sum(np.tanh(np.abs(self._Cs)), axis=2)
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

    def fit_transform(self, G, input_signal, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(input_signal, **kwargs)

    ## DB: I think that you should be able to call predict, i.e. KMeans
    ##     without needed to rerun `transform()`.
    def predict(self, input_signal=None, **kwargs):
        if not self._isfit:
            warnings.warn("Estimator is not fit. "
                          "Call MELDCluster.fit(). ")
            return None
        if self._C is None and s is None:
            warnings.warn("Estimator has no spectrogram to cluster. "
                          "Call MELDCluster.transform(s). ")
            return None
        # Checking if transform
        elif input_signal is not None and self._C is None:
            self.transform(input_signal, **kwargs)
        self.labels_ = self._clusterobj.fit_predict(self._C)
        self.labels_ = utils.sort_clusters_by_meld_score(self.labels_, self._input_signal)
        return self.labels_

    def fit_predict(self, G, input_signal, **kwargs):
        self.fit_transform(G, input_signal, **kwargs)
        return self.predict()

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, newk):
        self._n_clusters = newk
        self._clusterobj.set_params(n_clusters=self._n_clusters)

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop('k', False)
        if k:
            self._n_clusters = k
        self._sklearn_params = kwargs
        self._clusterobj.set_params(n_clusters=self._n_clusters, **kwargs)
