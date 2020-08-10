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
    D = np.random.normal(0, 2, (10, 2))
    G = gt.Graph(D, use_pygsp=False)
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
    EES = meld_op.fit_transform(data, labels, sample_idx=labels)
    meld.VertexFrequencyCluster().fit_transform(G=meld_op.graph, RES=labels, EES=EES)


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

    D = np.random.normal(0, 2, (1000, 2))
    RES = np.random.binomial(1, norm(D[:, 0]), 1000)

    meld_op = meld.MELD(
        verbose=0, knn=20, decay=10, thresh=0, anisotropy=0,
        filter=filter, solver="exact",
        normalize=False
    )
    B = meld_op.fit_transform(D, RES)

    if version.parse("1.17") <= version.parse(np.__version__) < version.parse("1.18"):
        if meld_op.filter == 'laplacian':
            np.testing.assert_allclose(np.sum(B), 519)
        else:
            np.testing.assert_allclose(np.sum(B), 519)
    else:
        if meld_op.filter == 'laplacian':
            np.testing.assert_allclose(np.sum(B), 532)
        else:
            np.testing.assert_allclose(np.sum(B), 532)

    # check changing filter params resets filter
    meld_op.set_params(beta=meld_op.beta + 1)
    assert meld_op.filt is None
    assert meld_op.EES is None

    meld_op.fit_transform(D, RES)
    assert meld_op.filt is not None
    assert meld_op.EES is not None

    # check changing graph params resets filter
    meld_op.set_params(knn=meld_op.knn + 1)
    assert meld_op.graph is None
    assert meld_op.filt is None
    assert meld_op.EES is None


def test_meld_invalid_lap_type():
    D = np.random.normal(0, 2, (1000, 2))
    # lap type TypeError
    lap_type = "hello world"
    with assert_raises_message(
        ValueError,
        "lap_type value {} not recognized. "
        "Choose from ['combinatorial', 'normalized']".format(lap_type),
    ):
        meld.MELD(verbose=0, lap_type=lap_type).fit(D)


def test_meld_res_wrong_shape():
    D = np.random.normal(0, 2, (100, 2))
    # RES wrong shape
    RES = np.ones([101, 2])
    with assert_raises_message(
        ValueError,
        "Input data ({}) and input graph ({}) "
        "are not of the same size".format(RES.shape, D.shape[0]),
    ):
        meld.MELD(verbose=0).fit_transform(
            X=D, RES=RES,
        )

def test_meld_res_dataframe():
    D = np.random.normal(0, 2, (100, 2))
    # RES wrong shape
    index = pd.Index(['cell_{}'.format(i) for i in range(100)])
    columns = pd.Index(['A'])
    RES = pd.DataFrame(np.ones([100, 1]),
            index=index,
            columns=columns)
    EES = meld.MELD(verbose=0).fit_transform(
        X=D, RES=RES,
    )
    assert np.all(EES.index == index)
    assert np.all(EES.columns == columns)

def test_meld_res_non_numeric():
    D = np.random.normal(size=(100,2))
    RES = np.random.choice(['A'], size=100)
    meld.MELD().fit_transform(D, RES)

    RES = np.random.choice(['A', 'B'], size=100)
    meld.MELD().fit_transform(D, RES)

    RES = np.random.choice(['A', 'B', 'C'], size=100)
    meld.MELD().fit_transform(D, RES)

class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # VertexFrequencyCluster
        # Custom window sizes
        self.window_sizes = np.array([2, 4, 8, 24])
        self.data, self.labels = make_batches(n_pts_per_cluster=100)
        meld_op = meld.MELD(verbose=0)
        self.EES = meld_op.fit_transform(
            self.data, sample_idx=self.labels, RES=self.labels
        )
        self.G = meld_op.graph

    def test_cluster(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_transform(self.G, RES=self.labels, EES=self.EES)
        # test sparse window
        for t in self.window_sizes:
            np.testing.assert_allclose(
                vfc_op._compute_window(self.G.diff_op, t).toarray(),
                vfc_op._compute_window(self.G.diff_op.toarray(), t),
            )
        # test sparse spectrogram
        for window in vfc_op.windows:
            np.testing.assert_allclose(
                vfc_op._compute_spectrogram(self.labels, window),
                vfc_op._compute_spectrogram(self.labels, sparse.csr_matrix(window)),
            )
        # test full sparse computation
        vfc_op.sparse = True
        sparse_spectrogram = vfc_op.fit_transform(self.G, RES=self.labels, EES=self.EES)
        assert sparse_spectrogram.shape == spectrogram.shape
        assert sparse.issparse(vfc_op._basewindow)

    def test_cluster_no_EES(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_predict(self.G, RES=self.labels, EES=None)

    def test_predit_setting_n_cluster(self):
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_transform(self.G, RES=self.labels, EES=None)
        clusters = vfc_op.predict(n_clusters=2)

    def test_2d(self):
        RES = np.array([self.labels, self.labels]).T
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        meld_op = meld.MELD(verbose=0,)
        EES = meld_op.fit_transform(self.data, RES=RES)
        clusters = vfc_op.fit_predict(self.G, RES=RES, EES=EES)
        assert len(clusters) == len(self.labels)

    def test_RES_EES_shape(self):
        RES = np.array([self.labels, self.labels]).T
        vfc_op = meld.VertexFrequencyCluster(window_sizes=self.window_sizes)
        meld_op = meld.MELD(verbose=0,)
        EES = meld_op.fit_transform(self.data, RES=RES)
        assert_raises_message(
            ValueError,
            "`RES` and `EES` must have the same shape."
            "Got RES: {} and EES: {}".format(str(RES[:, 1].shape), str(EES.shape)),
            vfc_op.fit_predict,
            G=self.G,
            RES=RES[:, 1],
            EES=EES,
        )

    def test_transform_before_fit(self):
        # Transform before fit
        assert_raises_message(
            ValueError,
            "Estimator must be `fit` before running `transform`.",
            meld.VertexFrequencyCluster().transform,
            RES=self.labels,
            EES=self.EES,
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

    def test_res_invalid(self):
        # RES not array-like
        assert_raises_message(
            TypeError,
            "`RES` must be array-like",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            RES="invalid",
            EES=self.EES,
        )

    def test_ees_invalid(self):
        # EES not array-like
        assert_raises_message(
            TypeError,
            "`EES` must be array-like",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            RES=self.labels,
            EES="invalid",
        )

    def test_res_wrong_length(self):
        # RES and n mismatch
        assert_raises_message(
            ValueError,
            "At least one axis of `RES` must be of length `N`.",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            RES=np.ones(7),
            EES=self.EES,
        )

    def test_ees_wrong_length(self):
        # EES and n mismatch
        assert_raises_message(
            ValueError,
            "At least one axis of `EES` must be of length `N`.",
            meld.VertexFrequencyCluster().fit_transform,
            G=self.G,
            RES=self.labels,
            EES=np.ones(7),
        )

    def test_set_params(self):
        # KMeans params
        vfc_op = meld.VertexFrequencyCluster()
        vfc_op.set_kmeans_params(n_clusters=2)
        assert vfc_op.n_clusters == 2

    def test_power_sparse(self):
        vfc_op = meld.VertexFrequencyCluster()
        vfc_op._power_matrix(self.G.diff_op, 2)
