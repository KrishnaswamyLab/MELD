# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
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


def normalize_EES_within_replicate(EES):
    '''
    Takes a two-column EES representing matched experimental and control
    pairs from the same replicate and normalizes the EES row-sum to 1.
    '''
    if EES.shape[1] != 2:
        raise ValueError(
            "It is currently only possible to normalize two samaples"
            "from a matched experimental and control pair. As such,"
            "EES.shape[1] must equal 2. Got {}".format(EES.shape[1])
        )
    return sklearn.preprocessing.normalize(EES, norm='l1')
