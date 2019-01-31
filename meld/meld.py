import numpy as np
import pygsp
import graphtools
import graphtools.base
import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from meld import utils
import sklearn.preprocessing as sklp
import warnings


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

class MELD(BaseEstimator):
    """ MELD operator for filtering signals over a graph.

    Parameters
    ----------
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
        'chebyshev' uses a chebyshev polynomial approximation of the corresponding filter
        'exact' uses the eigenvalue solution to the problem
    M : int, optional, Default: 50
        Order of chebyshev approximation to use.
    lap_type : ('combinatorial', 'normalized'), Default: 'combinatorial'
        The kind of Laplacian to calculate
    suppress : bool
        Suppress warnings
    """

    def __init__(self, beta=1, offset=0, order=2, solver='chebyshev', M=50,
             lap_type='combinatorial', suppress=False):

        self.suppress = suppress

        self.beta = beta
        self.offset = offset
        self.order = order
        self.solver = solver
        self.M = M
        self.lap_type = lap_type
        self.filt = None

    def fit(self, G):
        """ Builds the MELD filter over a graph `G`.

        Parameters
        ----------
        G : graphtools.Graph object
            Graph to perform data smoothing over.

        """
        # Make sure that the input graph is a valid pygsp graph
        try:
            _check_pygsp_graph(G)
        except TypeError:
            G = G.to_pygsp()


        if not self.lap_type in ('combinatorial', 'normalized'):
            raise TypeError("lap_type must be 'combinatorial'"
                            " or 'normalized'. Got: '{}'".format(self.lap_type))
        if G.lap_type != self.lap_type:
            warnings.warn(
                "Changing lap_type may require recomputing the Laplacian",
                RuntimeWarning)
            G.compute_laplacian(self.lap_type)

        # Generate MELD filter
        def filterfunc(x): return 1 / (1 + (self.beta * x - self.offset)**self.order)

        G.estimate_lmax()
        self.filt = pygsp.filters.Filter(G, filterfunc)  # build filter

    def transform(self, X, G):
        """ Filters a signal `X` over graph `G`.

        Parameters
        ----------
        X : ndarray [n, p]
            2 dimensional input signal array to filter.

        G : graphtools.Graph object
            Graph to perform data smoothing over.

        Returns
        -------
        X_nu : ndarray [n, p]
            Smoothed version of X

        """
        try:
            _check_pygsp_graph(G)
        except TypeError:
            G = G.to_pygsp()
        # Checking shape of X and G match
        if X.shape[0] != G.N:
            if len(X.shape) > 1 and X.shape[1] == G.N:
                warnings.warn(
                    "Input matrix is column-wise rather than row-wise. "
                    "transposing (output will be transposed)",
                    RuntimeWarning)
                X = X.T
            else:
                raise ValueError(
                    "Input data ({}) and input graph ({}) "
                    "are not of the same size".format(X.shape, G.N))

        X_nu = self.filt.filter(X, method=self.solver, order=self.M)  # apply filter

        #X_nu = utils.convert_to_same_format(X_nu, X)

        return X_nu

    def fit_transform(self, X, G):
        """ Builds the MELD filter over a graph `G` and filters a signal `X`.

        Parameters
        ----------
        X : ndarray [n, p]
            2 dimensional input signal array to filter.

        G : graphtools.Graph object
            Graph to perform data smoothing over.

        Returns
        -------
        X_nu : ndarray [n, p]
            Smoothed version of X

        """

        self.fit(G)
        self.X_nu = self.transform(X, G)
        return self.X_nu

class VertexFrequencyCluster(BaseEstimator):
    """Performs Vertex Frequency clustering for data given a
       raw experimental signal and enhanced experimental signal.

    Parameters
    ----------
    n_clusters : int, optional, default: 10
        The number of clusters to form.
    window_count : int, optional, default: 9
        Number of windows to use if window_sizes = None
    window_sizes : None, optional, default: None
        ndarray of integer window sizes to supply to t
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
    def __init__(self, n_clusters=10, window_count=9, window_sizes=None,
                 suppress=False, **kwargs):

        self.suppress = suppress
        self._basewindow = None
        if window_sizes is None:
            self.window_sizes = np.power(2, np.arange(window_count))
        else:
            self.window_sizes = window_sizes

        self.window_count = np.min(self.window_sizes.shape)
        self.n_clusters = n_clusters
        self.window = None
        self.eigenvectors = None
        self.N = None
        self.spec_hist = None
        self.spectrogram = None
        self.isfit = False
        self.X = None
        self._sklearn_params = kwargs
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

    def _compute_spectrogram(self, X, window):
        """_compute_spectrogram: computes spectrograms for
        arbitrary window/signal/graph combinations

        Parameters
        ----------
        X : np.ndarray
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
        ##  I can't tell, but for now I think that this can never happen
        ##  without the user deliberately making a mistake - DB
        #if sparse.issparse(window):
        #    window = window.toarray()
#
        #else:
        #    if not isinstance(window, np.ndarray):
        #        raise TypeError(
        #            "window must be a numpy.array or"
        #            "scipy.sparse.csr_matrix.")

        self.X = X
        C = np.multiply(window, self.X[:, None])
        C = sklp.normalize(self.eigenvectors.T@C, axis=0)
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
        assert not sparse.issparse(window)
        #if sparse.issparse(window):]
        #    window = window.toarray()
#
        #else:
        #    if not isinstance(window, np.ndarray):
        #        raise TypeError(
        #            "window must be a numpy.array or"
        #            "scipy.sparse.csr_matrix.")
        window = np.linalg.matrix_power(window, kwargs['t'])
        return sklp.normalize(window, 'l2', axis=0).T

    def fit(self, G):
        '''Sets eigenvectors and windows.'''

        _check_pygsp_graph(G)

        if self._basewindow is None:
            if sparse.issparse(G.diff_op):
                # TODO: support for sparse diffusion operator
                self._basewindow = G.diff_op.toarray()
            else:
                # Can this ever happen?
                self._basewindow = G.diff_op
        self.window = np.zeros((G.N, G.N, self.window_count))

        for i, t in enumerate(self.window_sizes):
            self.window[:, :, i] = self._compute_window(
                self._basewindow, t=t).astype(float)
        # Compute Fourier basis. This may take some time.
        G.compute_fourier_basis()
        self.eigenvectors = G.U
        self.N = G.N
        self.isfit = True
        return self

    def transform(self, X, center=True):
        self.X = X
        # DB - removing warnings the user can do nothing about
        #if not self.suppress:
        #    if (self.spectrogram is not None or self.spec_hist is not None):
        #        warnings.warn("Overwriting previous spectrogram. "
        #                      "Suppress this warning with "
        #                      "VertexFrequencyCluster(suppress=True)")
        if not self.isfit:
            raise ValueError(
                'Estimator must be `fit` before running `transform`.')

        else:
            if not isinstance(self.X, (list, tuple, np.ndarray)):
                raise TypeError('`X` must be array-like')
            self.X = np.array(self.X)
            if self.N not in self.X.shape:
                raise ValueError('At least one axis of X must be'
                                 ' of length N.')

            self.X = self.X - self.X.mean()
            self.spectrogram = np.zeros((self.N, self.N))
            self.spec_hist = np.zeros((
                self.N, self.N, self.window_count))
            for t in range(self.window_count):
                temp = self._compute_spectrogram(
                    self.X, self.window[:, :, t])
                # There's maybe something wrong here
                self.spec_hist[:, :, t] = temp
                # temp = self._activate(temp)
                # temp = sklp.normalize(temp, 'l2', axis=1)
                # This work goes nowhere

            self.spectrogram = np.sum(np.tanh(np.abs(self.spec_hist)), axis=2)
            """ This can be added later to support multiple signals
            for i in range(ncols):
                for t in range(self.window_count):
                    temp = self._compute_spectrogram(
                        s[:, i], self.eigenvectors, self.window[:, :, t])
                    if self._activated:
                        temp = self._activate(
                            temp)
                    if self._store_spec_hist:
                        self.spec_hist[:, :, t, i] = temp
                    else:
                        self.spectrogram[:, :, i] += temp"""

        return self.spectrogram

    def fit_transform(self, G, X, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(X, **kwargs)

    # DB: I think that you should be able to call predict, i.e. KMeans
    # without needed to rerun `transform()`.
    def predict(self, X=None, **kwargs):
        if not self.isfit:
            warnings.warn("Estimator is not fit. "
                          "Call VertexFrequencyCluster.fit(). ")
            return None
        if self.spectrogram is None and X is None:
            warnings.warn("Estimator has no spectrogram to cluster. "
                          "Call VertexFrequencyCluster.transform(s). ")
            return None
        # Checking if transform
        elif X is not None and self.spectrogram is None:
            self.transform(X, **kwargs)
        self.labels_ = self._clusterobj.fit_predict(self.spectrogram)
        self.labels_ = utils.sort_clusters_by_meld_score(
            self.labels_, self.X)
        return self.labels_

    def fit_predict(self, G, X, **kwargs):
        self.fit_transform(G, X, **kwargs)
        return self.predict()

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop('k', False)
        if k:
            self.n_clusters = k
        self._sklearn_params = kwargs
        self._clusterobj.set_params(n_clusters=self.n_clusters, **kwargs)
