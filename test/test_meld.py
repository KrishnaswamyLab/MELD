import numpy as np
import graphtools as gt
import meld
# from make_batches import make_batches


def make_batches(n_pts_per_cluster=5000):
    data = []
    labels = []

    make = lambda x, y, s: np.concatenate([np.random.normal(
        x, s, (n_pts_per_cluster, 1)), np.random.normal(y, s, (n_pts_per_cluster, 1))], axis=1)
    # batch 1
    d = [make(0, 0, .1), make(1, 1, .1), make(0, 1, .1)]
    l = np.zeros(len(d) * n_pts_per_cluster)
    d = np.concatenate(d, axis=0)

    data.append(d)
    labels.append(l)

    # batch 2
    d = [make(1, -1, .1), make(2, 0, .1), make(-2, -1, .1)]
    l = np.ones(len(d) * n_pts_per_cluster)
    d = np.concatenate(d, axis=0)

    data.append(d)
    labels.append(l)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels


def test_mnn():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    labels_meld = meld_op.fit_transform(labels, G)
    meld.VertexFrequencyCluster().fit_transform(G, labels)

def test_meld():
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
