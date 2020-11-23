# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
import graphtools as gt
import meld
import pygsp
import unittest

from scipy import sparse
from parameterized import parameterized
from utils import make_batches, assert_warns_message, assert_raises_message
from nose.tools import assert_raises

from packaging import version


def test_check_pygsp_graph():
    # _check_pygsp_graph
    data = np.random.normal(0, 2, (10, 2))
    G = gt.Graph(data, use_pygsp=False)
    assert isinstance(meld.utils._check_pygsp_graph(G), pygsp.graphs.Graph)
    assert_raises_message(
        TypeError,
        "Input graph should be of type graphtools.base.BaseGraph. "
        "With graphtools, use the `use_pygsp=True` flag.",
        meld.utils._check_pygsp_graph,
        G="hello world",
    )


def test_mnn():
    data, labels = make_batches(n_pts_per_cluster=250)
    meld_op = meld.MELD(verbose=0)
    sample_densities = meld_op.fit_transform(data, labels, sample_idx=labels)
    sample_likelihoods = meld.utils.normalize_densities(sample_densities)
    meld.VertexFrequencyCluster().fit_transform(
        G=meld_op.graph,
        sample_indicator=meld_op.sample_indicators['expt'],
        likelihood=sample_likelihoods['expt'])


@parameterized([("heat",), ("laplacian",)])
def test_meld(filter):
    # MELD operator
    # Numerical accuracy
    np.random.seed(42)

    def norm(x):
        x = x.copy()
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    data = np.random.normal(0, 2, (1000, 2))
    sample_labels = np.random.binomial(1, norm(data[:, 0]), 1000)
    sample_labels = np.array(['treat' if val else 'ctrl' for val in sample_labels])

    meld_op = meld.MELD(
        verbose=0, knn=20, decay=10, thresh=0, anisotropy=0,
        filter=filter, solver="exact",
        sample_normalize=False
    )
    densities = meld_op.fit_transform(data, sample_labels)
    expt_density = densities.iloc[:, 1]

    if version.parse("1.17") <= version.parse(np.__version__) < version.parse("1.18"):
        if meld_op.filter == 'laplacian':
            np.testing.assert_allclose(np.sum(expt_density), 519)
        else:
            np.testing.assert_allclose(np.sum(expt_density), 519)
    else:
        if meld_op.filter == 'laplacian':
            np.testing.assert_allclose(np.sum(expt_density), 532)
        else:
            np.testing.assert_allclose(np.sum(expt_density), 532)

    # check changing filter params resets filter
    meld_op.set_params(beta=meld_op.beta + 1)
    assert meld_op.sample_densities is None

    meld_op.fit_transform(data, sample_labels)
    assert meld_op.sample_densities is not None

    # check changing graph params resets filter
    meld_op.set_params(knn=meld_op.knn + 1)
    assert meld_op.graph is None
    assert meld_op.sample_densities is None


def test_meld_invalid_lap_type():
    data = np.random.normal(0, 2, (1000, 2))
    # lap type TypeError
    lap_type = "hello world"
    with assert_raises_message(
        ValueError,
        "lap_type value {} not recognized. "
        "Choose from ['combinatorial', 'normalized']".format(lap_type),
    ):
        meld.MELD(verbose=0, lap_type=lap_type).fit(data)


def test_meld_labels_wrong_shape():
    data = np.random.normal(0, 2, (100, 2))
    # sample_indicator wrong shape
    sample_labels = np.ones([101, 2], dtype=str)
    with assert_raises_message(
        ValueError,
        "Input data ({}) and input graph ({}) "
        "are not of the same size".format(sample_labels.shape, data.shape[0]),
    ):
        meld.MELD(verbose=0).fit_transform(
            X=data, sample_labels=sample_labels,
        )

def test_meld_label_2d():
    data = np.random.normal(0, 2, (100, 2))
    # Create a dataframe with a index
    index = pd.Index(['cell_{}'.format(i) for i in range(100)])
    columns = pd.Index(['A'])
    sample_labels = pd.DataFrame(
            np.concatenate([np.zeros((50,1)), np.ones((50,1))]),
            index=index,
            columns=columns,
            dtype=str,
            )
    meld_op = meld.MELD(verbose=0)

    sample_densities = meld_op.fit_transform(
            X=data, sample_labels=sample_labels,
            )


def test_meld_label_dataframe():
    data = np.random.normal(0, 2, (100, 2))
    # Create a dataframe with a index
    index = pd.Index(['cell_{}'.format(i) for i in range(100)])
    sample_labels = pd.DataFrame(
            np.concatenate([np.zeros(50), np.ones(50)]),
            index=index,
            columns=['sample_labels'],
            dtype=str)

    meld_op = meld.MELD(verbose=0)
    sample_densities = meld_op.fit_transform(
        X=data, sample_labels=sample_labels,
    )
    assert np.all(sample_densities.index == index)
    assert np.all(sample_densities.columns == pd.Index(np.unique(sample_labels)))

def test_meld_labels_non_numeric():
    data = np.random.normal(size=(100,2))

    sample_labels = np.random.choice(['A', 'B'], size=100)
    meld_op = meld.MELD()
    meld_op.fit_transform(data, sample_labels)

    sample_labels = np.random.choice(['A', 'B', 'C'], size=100)
    meld_op = meld.MELD()
    sample_densities = meld_op.fit_transform(data, sample_labels)
    assert np.all(sample_densities.columns == ['A', 'B', 'C'])

class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # VertexFrequencyCluster
        # Custom window sizes
        self.window_sizes = np.array([2, 4, 8, 24])
        self.data, self.sample_labels = make_batches(n_pts_per_cluster=100)
        meld_op = meld.MELD(verbose=0)
        self.densities = meld_op.fit_transform(
            self.data, sample_labels=self.sample_labels
        )
        self.sample_indicators = meld_op.sample_indicators
        self.likelihoods = meld.utils.normalize_densities(self.densities)
        self.G = meld_op.graph

    def test_cluster(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_transform(self.G,
                                sample_indicator=self.sample_indicators['expt'],
                                likelihood=self.likelihoods['expt'])
        # test sparse window
        for t in self.window_sizes:
            np.testing.assert_allclose(
                vfc_op._compute_window(self.G.diff_op, t).toarray(),
                vfc_op._compute_window(self.G.diff_op.toarray(), t),
            )
        # test sparse spectrogram
        for window in vfc_op.windows:
            np.testing.assert_allclose(
                vfc_op._compute_spectrogram(self.sample_indicators['expt'], window),
                vfc_op._compute_spectrogram(self.sample_indicators['expt'], sparse.csr_matrix(window)),
            )
        # test full sparse computation
        vfc_op.sparse = True
        sparse_spectrogram = vfc_op.fit_transform(self.G,
                                sample_indicator=self.sample_indicators['expt'],
                                likelihood=self.likelihoods['expt'])
        assert sparse_spectrogram.shape == spectrogram.shape
        assert sparse.issparse(vfc_op._basewindow)

        # test _compute_spectrogram wrong size signal
        with assert_raises_message(
            ValueError,
            "sample_indicator must be 1-dimensional. Got shape: {}".format(self.data.shape),
        ):
            vfc_op._compute_spectrogram(self.data, window)


    def test_cluster_no_likelihood(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_predict(
                    self.G,
                    sample_indicator=self.sample_indicators['expt'],
                    likelihood=self.likelihoods['expt'])

    def test_predit_setting_n_cluster(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_transform(
                    self.G,
                    sample_indicator=self.sample_indicators['expt'],
                    likelihood=self.likelihoods['expt'])
        clusters = vfc_op.predict(n_clusters=2)

    def test_2d(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        clusters = vfc_op.fit_predict(
                    self.G,
                    sample_indicator=self.sample_indicators['expt'],
                    likelihood=self.likelihoods['expt'])

        assert len(clusters) == len(self.sample_labels)

    def test_sample_indicator_likelihood_shape(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        meld_op = meld.MELD(verbose=0,)
        assert_raises_message(
            ValueError,
            "`sample_indicator` and `likelihood` must have the same shape. "
            "Got sample_indicator: {} and likelihood: {}".format(
                str(self.sample_indicators['expt'].shape),
                str(self.likelihoods.shape)
                ),
            vfc_op.fit_predict,
            G=self.G,
            sample_indicator=self.sample_indicators['expt'],
            likelihood=self.likelihoods,
        )

    def test_transform_before_fit(self):
        # Transform before fit
        assert_raises_message(
            ValueError,
            "Estimator must be `fit` before running `transform`.",
            meld.VertexFrequencyCluster().transform,
            sample_indicator=self.sample_indicators['expt'],
            likelihood=self.likelihoods['expt'],
        )

    def test_predict_before_fit(self):
        # predict before fit
        assert_raises_message(
            ValueError,
            "Estimator is not fit. Call VertexFrequencyCluster.fit().",
            meld.VertexFrequencyCluster().predict
        )

    def test_predict_before_transform(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        vfc_op.fit(self.G)
        # predict before transform
        assert_raises_message(
            ValueError,
            "Estimator is not transformed. " "Call VertexFrequencyCluster.transform().",
            vfc_op.predict
        )

    def test_sample_indicator invalid(self):
        # sample_indicator not array-like
        assert_raises_message(
            TypeError,
            "`sample_indicator` must be array-like",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            sample_indicator="invalid",
        )

    def test_likelihood_invalid(self):
        # likelihood not array-like
        assert_raises_message(
            TypeError,
            "`likelihood` must be array-like",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            sample_indicator=self.sample_indicators["expt"],
            likelihood="invalid",
        )

    def test_sample_indicator wrong_length(self):
        # sample_indicator and n mismatch
        assert_raises_message(
            ValueError,
            "At least one axis of `sample_indicator` must be of length `N`.",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            sample_indicator=np.ones(7),
            likelihood=self.likelihoods['expt'],
        )

    def test_likelihood_wrong_length(self):
        # likelihood and n mismatch
        assert_raises_message(
            ValueError,
            "At least one axis of `likelihood` must be of length `N`.",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            sample_indicator=self.sample_indicators["expt"],
            likelihood=np.ones(7),
        )

    def test_set_params(self):
        # KMeans params
        vfc_op = meld.VertexFrequencyCluster()
        vfc_op.set_kmeans_params(n_clusters=2)
        assert vfc_op.n_clusters == 2

    def test_power_sparse(self):
        vfc_op = meld.VertexFrequencyCluster()
        vfc_op._power_matrix(self.G.diff_op, 2)
