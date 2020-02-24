# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import numpy as np
import graphtools as gt
import meld


def make_batches(n_pts_per_cluster=5000):
    data = []
    labels = []

    make = lambda x, y, s: np.concatenate(
        [
            np.random.normal(x, s, (n_pts_per_cluster, 1)),
            np.random.normal(y, s, (n_pts_per_cluster, 1)),
        ],
        axis=1,
    )
    # batch 1
    d = [make(0, 0, 0.1), make(1, 1, 0.1), make(0, 1, 0.1)]
    l = np.zeros(len(d) * n_pts_per_cluster)
    d = np.concatenate(d, axis=0)

    data.append(d)
    labels.append(l)

    # batch 2
    d = [make(1, -1, 0.1), make(2, 0, 0.1), make(-2, -1, 0.1)]
    l = np.ones(len(d) * n_pts_per_cluster)
    d = np.concatenate(d, axis=0)

    data.append(d)
    labels.append(l)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels


def test_utils():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    EES = meld.MELD().fit_transform(G, labels)

    clusters = meld.VertexFrequencyCluster().fit_predict(G=G, RES=labels, EES=EES)
    meld.utils.sort_clusters_by_meld_score(clusters, EES)

    cmap = meld.utils.get_meld_cmap()
