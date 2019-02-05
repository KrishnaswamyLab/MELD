import numpy as np
import graphtools as gt
import meld
import pygsp

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

def test_cluster():
    # VertexFrequencyCluster
    # Custom window sizes
    window_sizes = np.array([2, 4, 8, 24])
    data, labels = make_batches(n_pts_per_cluster=100)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    EES = meld_op.fit_transform(G=G, RES=labels)
    vfc_op = meld.VertexFrequencyCluster(
        window_sizes=window_sizes)

    vfc_op.fit(G)
    # predict before transform
    assert_raise_message(ValueError,
                         "Estimator is not transformed. "
                         "Call VertexFrequencyCluster.transform().",
                         vfc_op.predict, RES=labels)
    spectrogram = vfc_op.fit_transform(G, RES=labels, EES=EES)
    assert isinstance(vfc_op._basewindow, np.ndarray)
    vfc_op.sparse = True
    np.testing.assert_allclose(spectrogram, vfc_op.fit_transform(G, RES=labels, EES=EES))
    assert sparse.issparse(vfc_op._basewindow)

    # Transform before fit
    assert_raise_message(ValueError,
                         'Estimator must be `fit` before running `transform`.',
                         meld.VertexFrequencyCluster().transform, RES=labels, EES=EES)

    # predict before fit
    assert_raise_message(ValueError,
                         "Estimator is not fit. Call VertexFrequencyCluster.fit().",
                         meld.VertexFrequencyCluster().predict, RES=labels, EES=EES)
    # predict before transform
    assert_raise_message(ValueError,
                         "Estimator is not transformed. "
                         "Call VertexFrequencyCluster.transform().",
                         meld.VertexFrequencyCluster().predict, RES=labels, EES=EES)

    # RES not array-like
    assert_raise_message(TypeError,
                         '`RES` and `EES` must be array-like',
                         meld.VertexFrequencyCluster().fit_transform, G=G,
                         RES='hello world', EES=EES)

    # RES and n mismatch
    assert_raise_message(ValueError,
                         'At least one axis of `RES` and `EES` must be'
                         ' of length `N`.',
                         meld.VertexFrequencyCluster().fit_transform, G=G,
                         RES=np.ones(7), EES=EES)

    # Predict
    meld.VertexFrequencyCluster().fit_predict(G=G, RES=labels, EES=EES)

    # KMeans params
    vfc_op = meld.VertexFrequencyCluster()
    vfc_op.set_kmeans_params(k=2)
