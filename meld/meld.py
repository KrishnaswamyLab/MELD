# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
import pygsp
import graphtools
import scprep.utils

from . import utils
from graphtools.estimator import GraphEstimator, attribute
from functools import partial


class MELD(GraphEstimator):
    """MELD operator for filtering signals over a graph.

    Parameters
    ----------
    beta : int, optional, Default: 60
        Amount of smoothing to apply. Default value of 60 determined through analysis
        of simulated data using Splatter (https://github.com/Oshlack/splatter).
    offset: float, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.
        Recommend using an eigenvalue from the graph based on the
        spectral distribution. Should be in interval [0,1]
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
    normalize : boolean, optional, Default: True
        If True, the RES is column normalized to sum to 1
    """

    # parameters
    beta = attribute("beta", default=40, on_set=graphtools.utils.check_positive)
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
        beta=60,
        offset=0,
        order=1,
        filter="heat",
        solver="chebyshev",
        chebyshev_order=50,
        lap_type="combinatorial",
        normalize=True,
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
        self.normalize = normalize

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

    def _RES_from_sample_labels(self):
        '''
        Helper function to take an array of non-numerics and produce an RES.
        '''
        self.sample_labels_ = self.RES
        self.samples = np.unique(self.RES)
        self._RES_cls = pd.DataFrame

        if self.samples.shape[0] == 1:
            # Only have one sample label (i.e. [A, A, A, A])
            self.RES = pd.DataFrame(np.ones(self.sample_labels_.shape[0]),
                                    columns=self.samples)
        elif self.samples.shape[0] == 2:
            # When there's two samples (i.e. [A, A, B, B])
            # LabelBinarizer doesn't work nicely with only two labels
            self.RES = pd.DataFrame([self.sample_labels_ == self.samples[0],
                            self.sample_labels_ == self.samples[1]],
                            dtype=int,
                            index=self.samples).T

        else:
            # We have more than two samples, use label binarizer.
            import sklearn
            self._LB = sklearn.preprocessing.LabelBinarizer()
            RES = self._LB.fit_transform(self.sample_labels_)
            self.RES = pd.DataFrame(RES, columns=self._LB.classes_)
        self._RES_columns = self.RES.columns

        return scprep.utils.toarray(self.RES)

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
        self.graph = utils._check_pygsp_graph(self.graph)

        if RES.shape[0] != self.graph.N:
            raise ValueError(
                "Input data ({}) and input graph ({}) "
                "are not of the same size".format(RES.shape, self.graph.N)
            )

        self._RES_cls = type(RES)
        self._RES_index = None
        self._RES_columns = None
        if isinstance(RES, pd.DataFrame):
            self._RES_index, self._RES_columns = RES.index, RES.columns

        self.RES = scprep.utils.toarray(RES)

        # Need to handle multiple cases for how the RES is passed
        # Option 1, a categorical / series / something like ['A', 'A', 'B', 'B']
        if not np.issubdtype(self.RES.dtype, np.number):
            # If we have non-numeric RES, then we should create an RES from it
            self.RES = self._RES_from_sample_labels()

        if self.normalize:
            self.RES = self.RES / self.RES.sum(axis=0)

        self.graph.estimate_lmax()

        # Generate MELD filter
        if self.filter.lower() == "laplacian":

            def filterfunc(x):
                return 1 / (
                    1 + (self.beta * np.abs(x / self.graph.lmax - self.offset)) ** self.order
                )

        elif self.filter.lower() == "heat":

            def filterfunc(x):
                return (
                    np.exp(-self.beta * np.abs(x / self.graph.lmax - self.offset) ** self.order)
                )

        else:
            raise NotImplementedError

        # build filter
        self.filt = pygsp.filters.Filter(self.graph, filterfunc)

        # apply filter
        self.EES = self.filt.filter(self.RES, method=self.solver, order=self.chebyshev_order)

        # normalize if only two samaples
        if len(self.EES.shape) > 1 and self.EES.shape[1] == 2:
            self.EES = utils.normalize_EES_within_replicate(self.EES)

        if self._RES_cls != np.ndarray:
            self.RES = self._RES_cls(self.RES)
            self.EES = self._RES_cls(self.EES)

        if isinstance(self.RES, pd.DataFrame):
            if self._RES_index is not None:
                self.RES.index, self.EES.index = self._RES_index, self._RES_index
            if self._RES_columns is not None:
                self.RES.columns, self.EES.columns = self._RES_columns, self._RES_columns

        return self.EES

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
