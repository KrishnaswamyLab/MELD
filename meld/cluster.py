import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings

from . import utils


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
    sparse : bool, optional, default: False
        Use sparse matrices. This is significantly slower,
        but will use less memory
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
                 sparse=False, suppress=False, **kwargs):

        self.suppress = suppress
        self.sparse = sparse
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
        """Activate spectrograms for clustering

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
        """Computes spectrograms for arbitrary window/signal/graph combinations

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
        # I can't tell, but for now I think that this can never happen
        # without the user deliberately making a mistake - DB
        # if sparse.issparse(window):
        #    window = window.toarray()
        #
        # else:
        #    if not isinstance(window, np.ndarray):
        #        raise TypeError(
        #            "window must be a numpy.array or"
        #            "scipy.sparse.csr_matrix.")

        self.X = X
        if sparse.issparse(window):
            C = window.multiply(self.X[:, None])
        else:
            C = np.multiply(window, self.X[:, None])
        C = preprocessing.normalize(self.eigenvectors.T@C, axis=0)
        return C.T

    def _compute_window(self, window, t=1):
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
            window = window ** t
        else:
            window = np.linalg.matrix_power(window, t)
        return preprocessing.normalize(window, 'l2', axis=0).T

    def fit(self, G):
        '''Sets eigenvectors and windows.'''

        G = utils._check_pygsp_graph(G)

        if self._basewindow is None:
            self._basewindow = G.diff_op
        if not self.sparse and sparse.issparse(self._basewindow):
            self._basewindow = self._basewindow.toarray()
        elif self.sparse and not sparse.issparse(self._basewindow):
            self._basewindow = sparse.csr_matrix(self._basewindow)

        self.windows = []

        for t in self.window_sizes:
            self.windows.append(self._compute_window(
                self._basewindow, t=t).astype(float))
        # Compute Fourier basis. This may take some time.
        G.compute_fourier_basis()
        self.eigenvectors = G.U
        self.N = G.N
        self.isfit = True
        return self

    def transform(self, X, center=True):
        self.X = X
        # DB - removing warnings the user can do nothing about
        # if not self.suppress:
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
            self.spec_hist = []
            for i in range(self.window_count):
                spectrogram = self._compute_spectrogram(
                    self.X, self.windows[i])
                # There's maybe something wrong here
                spectrogram = self._activate(spectrogram)
                self.spec_hist.append(spectrogram)
                # temp = preprocessing.normalize(temp, 'l2', axis=1)
                # This work goes nowhere

            self.spectrogram = sum(self.spec_hist)
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
