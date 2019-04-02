# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import warnings

import scprep

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
    random_state : int or None, optional (default: None)
        Random seed for clustering
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
                 sparse=False, suppress=False, random_state=None, **kwargs):
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
        self.ees = None
        self.res = None
        self._sklearn_params = kwargs
        self._clusterobj = KMeans(n_clusters=n_clusters,
                                  random_state=random_state, **kwargs)

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

    def _compute_spectrogram(self, RES, window):
        """Computes spectrograms for arbitrary window/signal/graph combinations

        Parameters
        ----------
        RES : np.ndarray
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
        if len(RES.shape) == 1:
            RES = RES[:, None]

        if sparse.issparse(window):
            # the next computation becomes dense - better to make dense now
            C = window.multiply(RES).toarray()
        else:
            C = np.multiply(window, RES)
        C = preprocessing.normalize(self.eigenvectors.T @ C, axis=0)
        return C.T


    def _compute_multiresolution_spectrogram(self, RES):
        ''' Compute multiresolution spectrogram by repeatedly calling
            _compute_spectrogram '''

        spectrogram = np.zeros((self.windows[0].shape[1],
                                self.eigenvectors.shape[1]))

        for window in self.windows:
            spectrogram = self._compute_spectrogram(
                RES, window)
            # There's maybe something wrong here
            spectrogram = self._activate(spectrogram)
            spectrogram += spectrogram
        return spectrogram


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

    def _concat_EES_to_spectrogram(self, weight):
        '''Concatenates the EES to the spectrogram for clustering'''

        data = decomposition.PCA(25).fit_transform(self.spectrogram)

        range_dim = np.max(np.max(data, axis=0) - np.min(data, axis=0))
        range_meld = np.max(self.EES) - np.min(self.EES)
        scale = (range_dim / range_meld) * weight
        data_nu = np.c_[data, (self.EES * scale)]
        return data_nu

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

    def transform(self, RES, EES=None, weight=1, center=True):
        '''Calculates the spectrogram of the graph using the RES'''
        self.RES = RES
        self.EES = EES
        if not self.isfit:
            raise ValueError(
                'Estimator must be `fit` before running `transform`.')

        if not isinstance(self.RES, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError('`RES` must be array-like.')

        if EES is not None and not isinstance(self.EES, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError('`EES` must be array-like.')

        self.RES = np.array(self.RES)
        if not self.N in self.RES.shape:
            raise ValueError('At least one axis of `RES` must be'
                             ' of length `N`.')

        if EES is not None:
            if  self.N not in self.EES.shape:
                raise ValueError('At least one axis of `EES` must be'
                                 ' of length `N`.')
            if EES.shape != RES.shape:
                raise ValueError('`RES` and `EES` must have the same shape.'
                'Got RES: {} and EES: {}'.format(str(RES.shape), str(EES.shape)))
            self.EES = np.array(self.EES)


        self.RES = self.RES - self.RES.mean()

        if len(self.RES.shape) == 1:
            self.spectrogram = self._compute_multiresolution_spectrogram(self.RES)
        else:
            spectrograms = []
            for i in range(self.RES.shape[1]):
                curr_RES = scprep.select.select_cols(self.RES, idx=i)
                spectrograms.append(self._compute_multiresolution_spectrogram(curr_RES))
            self.spectrogram = np.hstack(spectrograms)

        # Appending the EES to the spectrogram
        # TODO: is this a bad idea?
        if EES is not None:
            self.spectrogram = self._concat_EES_to_spectrogram(weight)

        return self.spectrogram

    def fit_transform(self, G, RES, EES=None, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(RES, EES, **kwargs)

    def predict(self, **kwargs):
        '''Runs KMeans on the spectrogram.'''
        if not self.isfit:
            raise ValueError("Estimator is not fit. "
                             "Call VertexFrequencyCluster.fit().")
        if self.spectrogram is None:
            raise ValueError("Estimator is not transformed. "
                             "Call VertexFrequencyCluster.transform().")
        self.labels_ = self._clusterobj.fit_predict(self.spectrogram)
        self.labels_ = utils.sort_clusters_by_meld_score(
            self.labels_, self.RES)
        return self.labels_

    def fit_predict(self, G, RES, EES=None, **kwargs):
        self.fit_transform(G, RES, EES, **kwargs)
        return self.predict()

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop('k', False)
        if k:
            self.n_clusters = k
        self._sklearn_params = kwargs
        self._clusterobj.set_params(n_clusters=self.n_clusters, **kwargs)
