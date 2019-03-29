# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import numpy as np
import graphtools.base
import graphtools
import pygsp


def _check_pygsp_graph(G):
    if isinstance(G, graphtools.base.BaseGraph):
        if not isinstance(G, pygsp.graphs.Graph):
            G = G.to_pygsp()
    else:
        raise TypeError(
            "Input graph should be of type graphtools.base.BaseGraph."
            " With graphtools, use the `use_pygsp=True` flag.")
    return G


def get_sorting_map(labels, EES):
    uniq_clusters = np.unique(labels)
    means = np.array([np.mean(EES[labels == cl]) for cl in uniq_clusters])
    new_clust_map = {curr_cl: i for i, curr_cl in enumerate(
        uniq_clusters[np.argsort(means)])}
    return new_clust_map


def sort_clusters_by_meld_score(clusters, EES):
    new_clust_map = get_sorting_map(clusters, EES)
    return np.array([new_clust_map[cl] for cl in clusters])
