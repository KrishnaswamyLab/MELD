import numpy as np
import graphtools as gt
import meld

from scipy import sparse

from sklearn.utils.testing import assert_warns_message, assert_raise_message, assert_raises

from utils import make_batches


def test_mnn():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    labels_meld = meld_op.fit_transform(labels, G)
    meld.VertexFrequencyCluster().fit_transform(G=G, X=labels)

def test_meld():
    ## _check_pygsp_graph
    D = np.random.normal(0, 2, (10,2))
    G = gt.Graph(D, use_pygsp=False)
    assert_raise_message(
        TypeError,
        "Input graph should be of type pygsp.graphs.Graph. "
        "With graphtools, use the `use_pygsp=True` flag.",
        meld.meld._check_pygsp_graph,
        G=G)
    assert_raise_message(
        TypeError,
        "Input graph should be of graphtools.base.BaseGraph."
        "With graphtools, use the `use_pygsp=True` flag.",
        meld.meld._check_pygsp_graph,
        G='hello world')

    ## MELD operator
    # Numerical accuracy
    np.random.seed(42)
    def norm(x):
        x = x.copy()
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    D = np.random.normal(0, 2, (1000,2))
    X = np.random.binomial(1, norm(D[:,0]), 1000)
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=True)

    meld_op = meld.MELD()
    B = meld_op.fit_transform(X, G)

    assert np.isclose(np.sum(B), 532.0001992193013)

    # Pygsp conversion
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=False)

    meld_op = meld.MELD()
    B = meld_op.fit_transform(X, G)
    assert np.isclose(np.sum(B), 532.0001992193013)

    # lap type TypeError
    lap_type='hello world'
    meld_op = meld.MELD(lap_type='hello world')
    assert_raise_message(
        TypeError,
        "lap_type must be 'combinatorial'"
        " or 'normalized'. Got: '{}'".format(lap_type),
        meld_op.fit,
        G=G)

    # lap reconversion warning
    meld_op = meld.MELD()
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=True, lap_type='normalized')
    assert_warns_message(
        RuntimeWarning,
        "Changing lap_type may require recomputing the Laplacian",
        meld_op.fit,
        G=G)

    # X transpose
    meld_op = meld.MELD()
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=True)
    assert_warns_message(
        RuntimeWarning,
        "Input matrix is column-wise rather than row-wise. "
        "transposing (output will be transposed)",
        meld_op.fit_transform,
        X=np.ones([2, G.N]),
        G=G)

    # X wrong shape
    X = np.ones([2, G.N + 100])
    meld_op = meld.MELD()
    G = gt.Graph(D, knn=20, decay=10, use_pygsp=True)
    assert_raise_message(
        ValueError,
        "Input data ({}) and input graph ({}) "
        "are not of the same size".format(X.shape, G.N),
        meld_op.fit_transform,
        X=X,
        G=G)

    ## VertexFrequencyCluster
    # Custom window sizes
    window_sizes = np.array([2,4,8,24])
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    labels_meld = meld_op.fit_transform(labels, G)
    meld.VertexFrequencyCluster(window_sizes=window_sizes).fit_transform(G, labels)

    # Transform before fit
    assert_raise_message(ValueError,
    'Estimator must be `fit` before running `transform`.',
    meld.VertexFrequencyCluster().transform, X=labels)

    # X not array-like
    assert_raise_message(TypeError,
    '`X` must be array-like',
    meld.VertexFrequencyCluster().fit_transform, G=G, X='hello world')

    # X and n mismatch
    assert_raise_message(ValueError,
    'At least one axis of X must be'
                     ' of length N.',
    meld.VertexFrequencyCluster().fit_transform, G=G, X=np.ones(7))

    # Predict
    meld.VertexFrequencyCluster().fit_predict(G=G, X=labels)

    # KMeans params
    vfc_op = meld.VertexFrequencyCluster()
    vfc_op.set_kmeans_params(k=2)

    assert_raises(AssertionError, vfc_op._compute_window, window=sparse.csr_matrix([1]))
