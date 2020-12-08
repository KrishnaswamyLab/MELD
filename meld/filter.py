import pygsp
import numpy as np


def filter(
    signal,
    graph,
    filter,
    beta,
    offset=0,
    order=1,
    solver="chebyshev",
    chebyshev_order=None,
):
    """Implements the MELD filter for sample-associated density estimation

    Parameters
    ----------
    signal: array-like
        Signal(s) to filter
    beta : int
        Amount of smoothing to apply. Default value of 60 determined through analysis
        of simulated data using Splatter (https://github.com/Oshlack/splatter).
    offset: float, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.
        Recommend using an eigenvalue from the graph based on the
        spectral distribution. Should be in interval [0,1]
    order: int, optional, Default: 1
        Falloff and smoothness of the filter.
        High order leads to square-like filters.
    solver : string, optional, Default: 'chebyshev'
        Method to solve convex problem.
        'chebyshev' uses a chebyshev polynomial approximation of the corresponding
        filter. 'exact' uses the eigenvalue solution to the problem
    chebyshev_order : int, optional, Default: 50
        Order of chebyshev approximation to use.
    """

    graph.estimate_lmax()

    # Generate MELD filter
    if filter.lower() == "laplacian":

        def filterfunc(x):
            return 1 / (1 + (beta * np.abs(x / graph.lmax - offset)) ** order)

    elif filter.lower() == "heat":

        def filterfunc(x):
            return np.exp(-beta * np.abs(x / graph.lmax - offset) ** order)

    else:
        raise NotImplementedError

    # build filter
    filt = pygsp.filters.Filter(graph, filterfunc)

    # apply filter
    densities = filt.filter(signal, method=solver, order=chebyshev_order)

    return densities
