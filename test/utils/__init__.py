# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import warnings
import numpy as np

def reset_warnings():
    warnings.resetwarnings()
    warnings.simplefilter("error")
    ignore_numpy_warning()


def ignore_numpy_warning():
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning,
        message="the matrix subclass is not the recommended way to represent "
        "matrices or deal with linear algebra ")
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message="future versions will not create a writeable array from "
        "broadcast_array. Set the writable flag explicitly to avoid this warning.")

reset_warnings()

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
