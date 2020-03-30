# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import numpy as np
import pygsp
import graphtools

from graphtools.estimator import GraphEstimator, attribute
from functools import partial


class MELD(GraphEstimator):
    """MELD operator for filtering signals over a graph.

    Parameters
    ----------
    beta : int
        Amount of smoothing to apply.
    offset: int, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.
        Recommend using an eigenvalue from the graph based on the
        spectral distribution.
    order: int, optional, Default: 1
        Falloff and smoothness of the filter.
        High order leads to square-like filters.
    Solver : string, optional, Default: 'chebyshev'
        Method to solve convex problem.
        'chebyshev' uses a chebyshev polynomial approximation of the corresponding filter
        'exact' uses the eigenvalue solution to the problem
    chebyshev_order : int, optional, Default: 50
        Order of chebyshev approximation to use.
    lap_type : ('combinatorial', 'normalized'), Default: 'combinatorial'
        The kind of Laplacian to calculate
    """

    # parameters
    beta = attribute("beta", default=1, on_set=graphtools.utils.check_positive)
    offset = attribute("offset", default=0)
    order = attribute("order", default=1)
    filter = attribute(
        "filter",
        default="heat",
        on_set=partial(graphtools.utils.check_in, ["heat", "laplacian"]),
    )
    solver = attribute(
        "solver",
        default="chebyshev",
        on_set=partial(graphtools.utils.check_in, ["chebyshev", "exact"]),
    )
    chebyshev_order = attribute(
        "chebyshev_order",
        default=30,
        on_set=[graphtools.utils.check_int, graphtools.utils.check_positive],
    )
    lap_type = attribute(
        "lap_type",
        default="combinatorial",
        on_set=partial(graphtools.utils.check_in, ["combinatorial", "normalized"]),
    )

    # stored attributes
    filt = attribute("filt")
    EES = attribute("EES", doc="Enhanced Experimental Signal (smoothed RES)")

    def __init__(
        self,
        beta=1,
        offset=0,
        order=1,
        filter="heat",
        solver="chebyshev",
        chebyshev_order=50,
        lap_type="combinatorial",
        anisotropy=1,
        n_landmark=None,
        **kwargs
    ):

        self.beta = beta
        self.offset = offset
        self.order = order
        self.solver = solver
        self.chebyshev_order = chebyshev_order
        self.lap_type = lap_type
        self.filter = filter

        kwargs["use_pygsp"] = True
        super().__init__(anisotropy=anisotropy, n_landmark=n_landmark, **kwargs)

    def _reset_graph(self):
        self._reset_filter()

    def _reset_filter(self):
        self.filt = None
        self.EES = None

    def set_params(self, **params):
        for p in [
            "beta",
            "offset",
            "order",
            "solver",
            "chebyshev_order",
            "lap_type",
            "filter",
        ]:
            if p in params and params[p] != getattr(self, p):
                self._reset_filter()
                setattr(self, p, params[p])
                del params[p]
        super().set_params(**params)

    def transform(self, RES):
        """Filters a signal `RES` over the data graph.

        Parameters
        ----------
        RES : ndarray [n, p]
            1- or 2-dimensional Raw Experimental Signal array to filter.

        Returns
        -------
        EES : ndarray [n, p]
            Enhanced Experimental Signal (smoothed RES)
        """
        if RES.shape[0] != self.graph.N:
            raise ValueError(
                "Input data ({}) and input graph ({}) "
                "are not of the same size".format(RES.shape, self.graph.N)
            )

        self.graph.estimate_lmax()

        # Generate MELD filter
        if self.filter.lower() == "laplacian":

            def filterfunc(x):
                return 1 / (
                    1 + (self.beta * x / self.graph.lmax - self.offset) ** self.order
                )

        elif self.filter.lower() == "heat":

            def filterfunc(x):
                return (
                    np.exp((-self.beta * x / self.graph.lmax) - self.offset)
                    ** self.order
                )

        else:
            raise NotImplementedError

        # build filter
        self.filt = pygsp.filters.Filter(self.graph, filterfunc)

        # apply filter
        EES = self.filt.filter(RES, method=self.solver, order=self.chebyshev_order)

        return EES

    def fit_transform(self, X, RES, **kwargs):
        """Builds the MELD filter over a graph built on data `X` and filters a signal `RES`.

        Parameters
        ----------

        X : array-like, shape=[n_samples, m_features]
            Data on which to build graph to perform data smoothing over.

        RES : array-like, shape=[n_samples, p_signals]
            1- or 2-dimensional Raw Experimental Signal array to filter.

        kwargs : additional arguments for graphtools.Graph

        Returns
        -------
        EES : ndarray, shape=[n_samples, p_signals]
            Enhanced Experimental Signal (smoothed RES)
        """
        self.fit(X, **kwargs)
        self.EES = self.transform(RES)
        return self.EES
