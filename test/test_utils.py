import numpy as np
import graphtools as gt
import meld

from sklearn.utils.testing import assert_warns_message, assert_raise_message

def test_utils():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    EES = meld_op.fit_transform(labels, G)

    clusters = meld.VertexFrequencyCluster().fit_transform(G=G, X=labels)
    meld.utils.sort_clusters_by_meld_score(clusters, EES)
