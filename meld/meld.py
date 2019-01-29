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
                            " or 'normalized. Got: {}'".format(lap_type))
        if G.lap_type != self.lap_type:
            warnings.warn(
                "Changing lap_type may require recomputing the Laplacian")
            G.compute_laplacian(lap_type)

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

        # Checking shape of X and G match
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

        X_nu = self.filt.filter(X, method=self.solver, order=self.M)  # apply filter

        X_nu = utils.convert_to_same_format(X_nu, X)

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
        self.input_signal = None
        self._sklearn_params = kwargs
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
        if self.eigenvectors is None:
            raise ValueError(
                'Estimator must be `fit` before running `_compute_spectrogram`.')

        # TODO: Why is this sparse if it's no supported?
        if sparse.issparse(window):
            warnings.warn("sparse windows not supported."
                          "Casting to np.ndarray.")
            window = window.toarray()

        else:
            if not isinstance(window, np.ndarray):
                raise TypeError(
                    "window must be a numpy.array or"
                    "scipy.sparse.csr_matrix.")

        self.input_signal = input_signal
        C = np.multiply(window, self.input_signal[:, None])
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
        # Would be nice if this didn't get called all the time
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
        if self.isfit and not refit:
            warnings.warn("Estimator is already fit. "
                          "Call VertexFrequencyCluster.fit(G,refit=True)"
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
        self.window = np.zeros((G.N, G.N, self.window_count))

        for i, t in enumerate(self.window_sizes):
            self.window[:, :, i] = self._compute_window(
                self._basewindow, t=t).astype(float)
        self.eigenvectors = G.U
        self.N = G.N
        self.isfit = True
        return self

    def transform(self, input_signal, center=True):
        self.input_signal = input_signal
        if not self.suppress:
            if (self.spectrogram is not None or self.spec_hist is not None):
                warnings.warn("Overwriting previous spectrogram. "
                              "Suppress this warning with "
                              "VertexFrequencyCluster(suppress=True)")
        if not self.isfit:
            if not self.suppress:
                warnings.warn("Estimator is not fit. "
                              "Call VertexFrequencyCluster.fit(). ")
            return None
        else:
            if not isinstance(input_signal, (list, tuple, np.ndarray)):
                raise TypeError('`input_signal` must be an array')
            input_signal = np.array(input_signal)
            if self.N not in input_signal.shape:
                raise ValueError('At least one axis of input_signal must be'
                                 ' of length N.')
            else:
                input_signal = input_signal - input_signal.mean()
                self.spectrogram = np.zeros((self.N, self.N))
                self.spec_hist = np.zeros((
                    self.N, self.N, self.window_count))
                for t in range(self.window_count):
                    temp = self._compute_spectrogram(
                        input_signal, self.window[:, :, t])
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

    def fit_transform(self, G, input_signal, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(input_signal, **kwargs)

    # DB: I think that you should be able to call predict, i.e. KMeans
    # without needed to rerun `transform()`.
    def predict(self, input_signal=None, **kwargs):
        if not self.isfit:
            warnings.warn("Estimator is not fit. "
                          "Call VertexFrequencyCluster.fit(). ")
            return None
        if self.spectrogram is None and input_signal is None:
            warnings.warn("Estimator has no spectrogram to cluster. "
                          "Call VertexFrequencyCluster.transform(s). ")
            return None
        # Checking if transform
        elif input_signal is not None and self.spectrogram is None:
            self.transform(input_signal, **kwargs)
        self.labels_ = self._clusterobj.fit_predict(self.spectrogram)
        self.labels_ = utils.sort_clusters_by_meld_score(
            self.labels_, self.input_signal)
        return self.labels_

    def fit_predict(self, G, input_signal, **kwargs):
        self.fit_transform(G, input_signal, **kwargs)
        return self.predict()

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop('k', False)
        if k:
            self.n_clusters = k
        self._sklearn_params = kwargs
        self._clusterobj.set_params(n_clusters=self.n_clusters, **kwargs)
