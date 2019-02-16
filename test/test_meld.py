import numpy as np
import graphtools as gt
import meld
import pygsp
import unittest

from scipy import sparse

from sklearn.utils.testing import assert_warns_message, assert_raise_message, assert_raises

from utils import make_batches


def test_mnn():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    EES = meld_op.fit_transform(labels, G)
    meld.VertexFrequencyCluster().fit_transform(G=G, RES=labels, EES=EES)


def test_check_pygsp_graph():
    # _check_pygsp_graph
    D = np.random.normal(0, 2, (10, 2))
    G = gt.Graph(D, use_pygsp=False)
    assert isinstance(meld.utils._check_pygsp_graph(G), pygsp.graphs.Graph)
    assert_raise_message(
        TypeError,
        "Input graph should be of type graphtools.base.BaseGraph. "
        "With graphtools, use the `use_pygsp=True` flag.",
        meld.utils._check_pygsp_graph,
        G='hello world')


def test_meld():
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
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=True)

    meld_op = meld.MELD()
    B = meld_op.fit_transform(RES, G)

    assert np.isclose(np.sum(B), 532.0001992193013)

    meld_op = meld.MELD()
    B = meld_op.fit_transform(RES, gt.Graph(
        D, knn=20, decay=10, use_pygsp=False))
    assert np.isclose(np.sum(B), 532.0001992193013)

    # lap type TypeError
    lap_type = 'hello world'
    assert_raise_message(
        TypeError,
        "lap_type must be 'combinatorial'"
        " or 'normalized'. Got: '{}'".format(lap_type),
        meld.MELD(lap_type=lap_type).fit,
        G=G)

    # RES transpose
    assert_warns_message(
        RuntimeWarning,
        "Input matrix is column-wise rather than row-wise. "
        "transposing (output will be transposed)",
        meld_op.fit_transform,
        RES=np.ones([2, G.N]),
        G=G)

    # RES wrong shape
    RES = np.ones([2, G.N + 100])
    assert_raise_message(
        ValueError,
        "Input data ({}) and input graph ({}) "
        "are not of the same size".format(RES.shape, G.N),
        meld_op.fit_transform,
        RES=RES,
        G=G)

    # lap reconversion warning
    assert_warns_message(
        RuntimeWarning,
        "Changing lap_type may require recomputing the Laplacian",
        meld_op.fit,
        G=gt.Graph(D, knn=20, decay=10, use_pygsp=True, lap_type='normalized'))


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # VertexFrequencyCluster
        # Custom window sizes
        self.window_sizes = np.array([2, 4, 8, 24])
        data, self.labels = make_batches(n_pts_per_cluster=100)
        self.G = gt.Graph(data, sample_idx=self.labels, use_pygsp=True)
        meld_op = meld.MELD()
        self.EES = meld_op.fit_transform(G=self.G, RES=self.labels)

    def test_cluster(self):
        vfc_op = meld.VertexFrequencyCluster(
            window_sizes=self.window_sizes)
        spectrogram = vfc_op.fit_transform(
            self.G, RES=self.labels, EES=self.EES)
        # test sparse window
        for t in self.window_sizes:
            np.testing.assert_allclose(
                vfc_op._compute_window(self.G.diff_op, t).toarray(),
                vfc_op._compute_window(self.G.diff_op.toarray(), t))
        # test sparse spectrogram
        for window in vfc_op.windows:
            np.testing.assert_allclose(
                vfc_op._compute_spectrogram(self.labels, window),
                vfc_op._compute_spectrogram(self.labels, sparse.csr_matrix(window)))
        # test full sparse computation
        vfc_op.sparse = True
        sparse_spectrogram = vfc_op.fit_transform(
            self.G, RES=self.labels, EES=self.EES)
        assert sparse_spectrogram.shape == spectrogram.shape
        assert sparse.issparse(vfc_op._basewindow)

    def test_transform_before_fit(self):
        # Transform before fit
        assert_raise_message(ValueError,
                             'Estimator must be `fit` before running `transform`.',
                             meld.VertexFrequencyCluster().transform,
                             RES=self.labels, EES=self.EES)

    def test_predict_before_fit(self):
        # predict before fit
        assert_raise_message(ValueError,
                             "Estimator is not fit. Call VertexFrequencyCluster.fit().",
                             meld.VertexFrequencyCluster().predict,
                             RES=self.labels, EES=self.EES)

    def test_predict_before_transform(self):
        vfc_op = meld.VertexFrequencyCluster(
            window_sizes=self.window_sizes)
        vfc_op.fit(self.G)
        # predict before transform
        assert_raise_message(ValueError,
                             "Estimator is not transformed. "
                             "Call VertexFrequencyCluster.transform().",
                             vfc_op.predict, RES=self.labels)

    def test_res_invalid(self):
        # RES not array-like
        assert_raise_message(TypeError,
                             '`RES` must be array-like',
                             meld.VertexFrequencyCluster().fit_transform,
                             G=self.G, RES='invalid', EES=self.EES)

    def test_ees_invalid(self):
        # EES not array-like
        assert_raise_message(TypeError,
                             '`EES` must be array-like',
                             meld.VertexFrequencyCluster().fit_transform,
                             G=self.G, RES=self.labels, EES='invalid')

    def test_res_wrong_length(self):
        # RES and n mismatch
        assert_raise_message(ValueError,
                             'At least one axis of `RES` and `EES` must be'
                             ' of length `N`.',
                             meld.VertexFrequencyCluster().fit_transform,
                             G=self.G, RES=np.ones(7), EES=self.EES)

    def test_ees_wrong_length(self):
        # EES and n mismatch
        assert_raise_message(ValueError,
                             'At least one axis of `EES` must be'
                             ' of length `N`.',
                             meld.VertexFrequencyCluster().fit_transform,
                             G=self.G, RES=self.labels, EES=np.ones(7))

    def test_set_params(self):
        # KMeans params
        vfc_op = meld.VertexFrequencyCluster()
        vfc_op.set_kmeans_params(k=2)
        assert vfc_op._clusterobj.n_clusters == 2
