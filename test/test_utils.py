# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import graphtools as gt
import meld
from utils import make_batches


def test_utils():
    data, labels = make_batches(n_pts_per_cluster=250)
    G = gt.Graph(data, sample_idx=labels, use_pygsp=True)
    meld_op = meld.MELD()
    sample_densities = meld_op.fit_transform(G, labels)
    sample_likelihoods = meld.utils.normalize_densities(sample_densities)

    meld.VertexFrequencyCluster().fit_predict(
        G=G,
        sample_indicator=meld_op.sample_indicators["expt"],
        likelihood=sample_likelihoods["expt"],
    )

    meld.utils.get_meld_cmap()

    # Test normalize_densities
    # Three samples
    densities = np.ones([100, 3])
    meld.utils.normalize_densities(sample_densities=densities)

    # Two samples
    densities = np.ones([100, 2])
    meld.utils.normalize_densities(sample_densities=densities)
