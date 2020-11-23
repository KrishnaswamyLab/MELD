# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
import graphtools
import scprep.utils

from . import utils
from . import filter
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
    solver : string, optional, Default: 'chebyshev'
        Method to solve convex problem.
        'chebyshev' uses a chebyshev polynomial approximation of the corresponding filter
        'exact' uses the eigenvalue solution to the problem
    chebyshev_order : int, optional, Default: 50
        Order of chebyshev approximation to use.
    lap_type : ('combinatorial', 'normalized'), Default: 'combinatorial'
        The kind of Laplacian to calculate
    sample_normalize : boolean, optional, Default: True
        If True, the sample indicator vectors are column normalized to sum to 1
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
    sample_densities = attribute(
        "sample_densities",
        doc="Density associated with each sample")

    def __init__(
        self,
        beta=60,
        offset=0,
        order=1,
        filter="heat",
        solver="chebyshev",
        chebyshev_order=50,
        lap_type="combinatorial",
        sample_normalize=True,
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
        self.sample_normalize = sample_normalize

        kwargs["use_pygsp"] = True
        super().__init__(anisotropy=anisotropy, n_landmark=n_landmark, **kwargs)

    def _reset_graph(self):
        self._reset_filter()

    def _reset_filter(self):
        self.filt = None
        self.sample_densities = None

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

    def _create_sample_indicators(self, sample_labels):
        '''
        Helper function to take an array-like of non-numerics and produce a collection of sample indicator vectors.
        '''

        self.sample_labels_ = sample_labels
        self.samples = np.unique(sample_labels)


        try:
            labels = sample_labels.values
        except AttributeError:
            labels = self.sample_labels_

        if len(labels.shape) > 1:
            # If you have a 2D array
            if labels.shape[1] == 1:
                # If it's just a column-vector, reshape it
                labels = labels.reshape(-1)
            else:
                # If its got multiple-columns, raise Error
                raise ValueError(
                    "sample_labels must be a single column. Got"    "shape={}".format(labels.shape)
                    )


        if self.samples.shape[0] == 2:
            # When there's two samples (i.e. [A, A, B, B])
            # LabelBinarizer doesn't work nicely with only two labels
            # This creates a two-column dataframe using the sample labels
            df = pd.DataFrame(
                [
                labels == self.samples[0],
                labels == self.samples[1]
                ],
                columns=self._labels_index).astype(int)
            df.index = self.samples

            self.sample_indicators = df.T

        else:
            # We have more than two samples, use label binarizer.
            import sklearn
            self._LB = sklearn.preprocessing.LabelBinarizer()
            sample_indicators = self._LB.fit_transform(self.sample_labels_)
            self.sample_indicators = pd.DataFrame(sample_indicators, columns=self._LB.classes_)

        return self.sample_indicators

    def transform(self, sample_labels):
        """Filters a collection of sample_indicators over the data graph.

        Parameters
        ----------
        sample_indicators : ndarray [n, p]
            1- or 2-dimensional sample indicator array to filter.

        Returns
        -------
        sample_densities: ndarray [n, p]
            A density estimate for each sample.
        """
        self.graph = utils._check_pygsp_graph(self.graph)
        self._sample_labels = sample_labels

        if sample_labels.shape[0] != self.graph.N:
            raise ValueError(
                "Input data ({}) and input graph ({}) "
                "are not of the same size".format(sample_labels.shape, self.graph.N)
            )

        if len(np.unique(sample_labels)) == 1:
            raise ValueError(
                "Found only one unqiue sample label. Cannot estimate density "
                "of a single sample."
            )

        #self._label_cls = type(sample_labels)
        if isinstance(sample_labels, pd.DataFrame):
            self._labels_index = sample_labels.index
        else:
            self._labels_index = None

        self._create_sample_indicators(sample_labels)

        if self.sample_normalize:
            self.sample_indicators = self.sample_indicators / self.sample_indicators.sum(axis=0)

        # apply filter
        densities = filter.filter(signal=self.sample_indicators,
                                  graph=self.graph,
                                  filter=self.filter,
                                  beta=self.beta,
                                  offset=self.offset,
                                  order=self.order,
                                  solver=self.solver,
                                  chebyshev_order=self.chebyshev_order)

        self.sample_densities = pd.DataFrame(densities, index=self._labels_index, columns=self.sample_indicators.columns)

        return self.sample_densities

    def fit_transform(self, X, sample_labels, **kwargs):
        """Builds the MELD filter over a graph built on data `X` and estimates density of each sample in `sample_labels`

        Parameters
        ----------

        X : array-like, shape=[n_samples, m_features]
            Data on which to build graph to perform data smoothing over.

        sample_labels : array-like, shape=[n_samples, p_signals]
            1- or 2-dimensional array of non-numerics indicating the sample origin for each cell.

        kwargs : additional arguments for graphtools.Graph

        Returns
        -------
        sample_densities : ndarray, shape=[n_samples, p_signals]
            Density estimate for each sample over a graph built from X
        """
        self.fit(X, **kwargs)
        return self.transform(sample_labels)
