# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import pandas as pd
import graphtools.base
import graphtools
import pygsp
import scprep
import sklearn


def _check_pygsp_graph(G):
    if isinstance(G, graphtools.base.BaseGraph):
        if not isinstance(G, pygsp.graphs.Graph):
            G = G.to_pygsp()
    else:
        raise TypeError(
            "Input graph should be of type graphtools.base.BaseGraph."
            " With graphtools, use the `use_pygsp=True` flag."
        )
    return G


def get_meld_cmap():
    """Returns cmap used in publication for displaying EES.
    Inspired by cmocean `balance` cmap"""
    base_colors = [
        [0.22107637, 0.53245276, 0.72819301, 1.0],
        [0.7, 0.7, 0.7, 1],
        [0.75013244, 0.3420382, 0.22753009, 1.0],
    ]

    return scprep.plot.tools.create_colormap(base_colors)


def normalize_densities(sample_densities):
    """
    Takes a 2-d array of sample densities from the same replicate and
    normalizes the row-sum to 1.
    """
    if isinstance(sample_densities, pd.DataFrame):
        index, columns = sample_densities.index, sample_densities.columns

    norm_densities = sklearn.preprocessing.normalize(sample_densities, norm="l1")

    if isinstance(sample_densities, pd.DataFrame):
        norm_densities = pd.DataFrame(norm_densities, index=index, columns=columns)
    return norm_densities
