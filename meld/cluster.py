# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import warnings

import scprep

import time

from . import utils


class VertexFrequencyCluster(BaseEstimator):
    """Performs Vertex Frequency clustering for data given a
       raw experimental signal and enhanced experimental signal.

    Parameters
    ----------
    n_clusters : int, optional, default: 10
        The number of clusters to form.
    likelihood_bias : float, optional, default: 1
        A normalization term that biases clustering towards the
        likelihood (higher values) or towards the spectrogram (lower values)
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

    def __init__(
        self,
        n_clusters=10,
        likelihood_bias=1,
        window_count=9,
        window_sizes=None,
        sparse=False,
        suppress=False,
        random_state=None,
        **kwargs
    ):
        self.suppress = suppress
        self.sparse = sparse
        self._basewindow = None
        if window_sizes is None:
            self.window_sizes = np.power(2, np.arange(window_count))
        else:
            self.window_sizes = window_sizes

        self.window_count = np.min(self.window_sizes.shape)
        self.n_clusters = n_clusters
        self.likelihood_bias = likelihood_bias
        self.window = None
        self.eigenvectors = None
        self.N = None
        self.spec_hist = None
        self.spectrogram = None
        self.combined_spectrogram = None
        self.isfit = False
        self.likelihood = None
        self.sample_indicator = None
        self._sklearn_params = kwargs

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

    def _compute_spectrogram(self, sample_indicator, window):
        """Computes spectrograms for arbitrary window/signal/graph combinations

        Parameters
        ----------
        sample_indicator : np.ndarray
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
        tic = time.time()
        # print('    Computing spectrogram for window')
        if len(sample_indicator.shape) == 1:
            sample_indicator = np.array(sample_indicator)
        else:
            raise ValueError('sample_indicator must be 1-dimensional. Got shape: {}'.format(sample_indicator.shape))
        if sparse.issparse(window):
            # the next computation becomes dense - better to make dense now
            C = window.multiply(sample_indicator).toarray()
        else:
            C = np.multiply(window, sample_indicator)
        C = preprocessing.normalize(self.eigenvectors.T @ C, axis=0)
        # print('     finished in {:.2f} seconds'.format(time.time() - tic))
        return C.T

    def _compute_multiresolution_spectrogram(self, sample_indicator):
        """ Compute multiresolution spectrogram by repeatedly calling
            _compute_spectrogram """

        # spectrogram = np.zeros((self.windows[0].shape[1],
        #                        self.eigenvectors.shape[1]))

        # for window in self.windows:
        #    curr_spectrogram = self._compute_spectrogram(
        #        sample_indicator, window)
        #    curr_spectrogram = self._activate(curr_spectrogram)
        #    spectrogram += curr_spectrogram
        tic = time.time()
        # print('  Computing multiresolution spectrogram')
        spectrogram = np.zeros((self.windows[0].shape[1], self.eigenvectors.shape[1]))

        for window in self.windows:
            curr_spectrogram = self._compute_spectrogram(sample_indicator=sample_indicator, window=window)
            curr_spectrogram = self._activate(curr_spectrogram)
            spectrogram += curr_spectrogram

        # print('   finished in {:.2f} seconds'.format(time.time() - tic))

        return spectrogram

    def _compute_window(self, window, t=1):
        """_compute_window
        These windows mask the signal (sample_indicator) to perform a Windowed Graph
        Fourier Transform (WGFT) as described by Shuman et al.
        (https://arxiv.org/abs/1307.5708).

        This function is used when the power of windows is NOT diadic
        """
        if sparse.issparse(window):
            window = window ** t
        else:
            window = np.linalg.matrix_power(window, t)
        return preprocessing.normalize(window, "l2", axis=0).T

    def _power_matrix(self, a, n):
        if sparse.issparse(a):
            a = a ** n
        else:
            a = np.linalg.matrix_power(a, n)
        return a

    def _compute_windows(self):
        """_compute_window
        These windows mask the signal (sample_indicator) to perform a Windowed Graph
        Fourier Transform (WGFT) as described by Shuman et al.
        (https://arxiv.org/abs/1307.5708).

        This function is used when the power of windows is diadic and
        computes all windows efficiently.
        """
        windows = []
        curr_window = self._basewindow
        windows.append(preprocessing.normalize(curr_window, "l2", axis=0).T)
        for i in range(len(self.window_sizes) - 1):
            curr_window = self._power_matrix(curr_window, 2)
            windows.append(preprocessing.normalize(curr_window, "l2", axis=0).T)
        return windows

    def _combine_spectrogram_likelihood(self, spectrogram, likelihood):
        """Normalizes and concatenates the likelihood to the
           spectrogram for clustering"""

        spectrogram_n = spectrogram / np.linalg.norm(spectrogram)

        ees_n = likelihood / np.linalg.norm(likelihood, ord=2, axis=0)
        ees_n = ees_n * self.likelihood_bias
        data_nu = np.c_[spectrogram_n, ees_n]
        return data_nu

    def fit(self, G):
        """Sets eigenvectors and windows."""

        self.graph = utils._check_pygsp_graph(G)

        if self._basewindow is None:
            self._basewindow = G.diff_op
        if not self.sparse and sparse.issparse(self._basewindow):
            self._basewindow = self._basewindow.toarray()
        elif self.sparse and not sparse.issparse(self._basewindow):
            self._basewindow = sparse.csr_matrix(self._basewindow)

        self.windows = []
        tic = time.time()
        # print('Building windows')
        # Check if windows were generated using powers of 2
        if np.all(np.diff(np.log2(self.window_sizes)) == 1):
            self.windows = self._compute_windows()
        else:
            for t in self.window_sizes:
                self.windows.append(
                    self._compute_window(self._basewindow, t=t).astype(float)
                )
        # print(' finished in {:.2f} seconds'.format(time.time() - tic))

        tic = time.time()
        # print('Computing Fourier basis')
        # Compute Fourier basis. This may take some time.
        self.graph.compute_fourier_basis()
        self.eigenvectors = self.graph.U
        self.N = self.graph.N
        self.isfit = True
        # print(' finished in {:.2f} seconds'.format(time.time() - tic))

        return self

    def transform(self, sample_indicator, likelihood=None, center=True):
        """Calculates the spectrogram of the graph using the sample_indicator"""
        self.sample_indicator = sample_indicator
        self.likelihood = likelihood
        if not self.isfit:
            raise ValueError("Estimator must be `fit` before running `transform`.")

        if not isinstance(self.sample_indicator, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("`sample_indicator` must be array-like.")

        if likelihood is not None and not isinstance(
            self.likelihood, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)
        ):
            raise TypeError("`likelihood` must be array-like.")

        # Checking shape of sample_indicator
        self.sample_indicator = np.array(self.sample_indicator)
        if not self.N in self.sample_indicator.shape:
            raise ValueError("At least one axis of `sample_indicator` must be" " of length `N`.")

        # Checking shape of likelihood
        if likelihood is not None:
            if self.N not in self.likelihood.shape:
                raise ValueError("At least one axis of `likelihood` must be" " of length `N`.")
            if likelihood.shape != sample_indicator.shape:
                raise ValueError(
                    "`sample_indicator` and `likelihood` must have the same shape. "
                    "Got sample_indicator: {} and likelihood: {}".format(str(sample_indicator.shape), str(likelihood.shape))
                )
            self.likelihood = np.array(self.likelihood)

        # Subtract the mean from the sample_indicator
        if center:
            self.sample_indicator = self.sample_indicator - self.sample_indicator.mean()

        # If only one sample_indicator, no need to collect
        if len(self.sample_indicator.shape) == 1:
            self.spectrogram = self._compute_multiresolution_spectrogram(self.sample_indicator)
        else:
            # Create a list of spectrograms and concatenate them
            spectrograms = []
            for i in range(self.sample_indicator.shape[1]):
                curr_sample_indicator = scprep.select.select_cols(self.sample_indicator, idx=i)
                spectrograms.append(self._compute_multiresolution_spectrogram(curr_sample_indicator))
            self.spectrogram = np.hstack(spectrograms)

        # Appending the likelihood to the spectrogram
        if self.likelihood is not None:
            self.combined_spectrogram = self._combine_spectrogram_likelihood(
                spectrogram=self.spectrogram, likelihood=self.likelihood
            )

        return self.spectrogram

    def fit_transform(self, G, sample_indicator, likelihood=None, **kwargs):
        self.fit(G, **kwargs)
        return self.transform(sample_indicator, likelihood, **kwargs)

    def predict(self, n_clusters=None, **kwargs):
        """Runs KMeans on the spectrogram."""

        if n_clusters is not None:
            self.n_clusters = n_clusters
        self._clusterobj = KMeans(n_clusters=self.n_clusters, **kwargs, **self._sklearn_params)

        if not self.isfit:
            raise ValueError(
                "Estimator is not fit. " "Call VertexFrequencyCluster.fit()."
            )
        if self.spectrogram is None:
            raise ValueError(
                "Estimator is not transformed. "
                "Call VertexFrequencyCluster.transform()."
            )

        if self.combined_spectrogram is None:
            data = self.spectrogram
        else:
            data = self.combined_spectrogram
        tic = time.time()
        # print('Running PCA on the spectrogram')
        data = decomposition.PCA(self.n_clusters).fit_transform(data)
        # print(' finished in {:.2f} seconds'.format(time.time()-tic))

        tic = time.time()
        # print('Running clustering')
        self.labels_ = self._clusterobj.fit_predict(data)

        self.labels_ = scprep.utils.sort_clusters_by_values(self.labels_, self.likelihood)
        # print(' finished in {:.2f} seconds'.format(time.time()-tic))

        return self.labels_

    def fit_predict(self, G, sample_indicator, likelihood=None, **kwargs):
        self.fit_transform(G, sample_indicator, likelihood, **kwargs)
        return self.predict()

    def set_kmeans_params(self, **kwargs):
        k = kwargs.pop("n_clusters", False)
        if k:
            self.n_clusters = k
        self._sklearn_params = kwargs
