# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import graphtools as gt
import meld
from utils import make_batches, assert_warns_message, assert_raises_message

def test_utils():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    sample_densities = meld_op.fit_transform(G, labels)
    sample_likelihoods = meld.utils.normalize_densities(sample_densities)

    clusters = meld.VertexFrequencyCluster().fit_predict(
        G=G,
        sample_indicator=meld_op.sample_indicators['expt'],
        likelihood=sample_likelihoods['expt'])

    cmap = meld.utils.get_meld_cmap()

    # Test sample normalization
    D = np.random.normal(0, 2, (100, 2))
    # Three samples
    densities = np.ones([100, 3])
    meld.utils.normalize_densities(sample_densities=densities)


    # Two samples
    densities = np.ones([100, 2])
    meld.utils.normalize_densities(sample_densities=densities)
