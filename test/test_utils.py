# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import graphtools as gt
import meld
from utils import make_batches, assert_warns_message, assert_raises_message

def test_utils():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    EES = meld.MELD().fit_transform(G, labels)

    clusters = meld.VertexFrequencyCluster().fit_predict(G=G, RES=labels, EES=EES)

    cmap = meld.utils.get_meld_cmap()

    # Test sample normalization
    D = np.random.normal(0, 2, (100, 2))
    # EES wrong shape
    EES = np.ones([100, 3])
    with assert_raises_message(
        ValueError,
            "It is currently only possible to normalize two samaples"
            "from a matched experimental and control pair. As such,"
            "EES.shape[1] must equal 2. Got {}".format(EES.shape[1])
        ):
        meld.utils.normalize_EES_within_replicate(EES)

    # EES correct shape
    EES = np.ones([100, 2])
    meld.utils.normalize_EES_within_replicate(EES=EES)
