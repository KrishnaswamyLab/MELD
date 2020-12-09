# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt


class Benchmarker(object):
    """Creates random signals over a dataset for benchmarking.

    Results are used for quantitative comparisons and for parameter searches
    on a specific dataset.

    Parameters
    ----------
    seed : integer or numpy.RandomState, optional, default: None
        Random state. Defaults to the global `numpy` random number generator

    Attributes
    ----------
    data_phate : array, shape=[n_samples, 3]
        Embedding of the data used to create random signals
    pdf : array, shape=[n_samples]
        Ground truth probability density function created over the input data.
    sample_indicator : array, shape=[n_samples]
        Sample indicator vectors used for k-NN and graph averaging.
    sample_labels : array, shape=[n_samples, 2]
        Indicates the sample to which each cell is assigned.
    graph : graphtools.base.BaseGraph
        The graph built on the input data
    graph_kNN : graphtools.graphs.kNNGraph
        The graph built on the input data
    meld_op : meld.meld.MELD
        MELD operator used to derive sample-associated density estimates
    sample_likelihoods : array, shape=[n_samples, 2]
        The relative likelihood that each cell originally sampled from either condition.
        Should converge to Benchmarker.pdf

    """

    def __init__(self, seed=None):
        self.seed = seed
        self.data_phate = None
        self.pdf = None
        self.sample_indicator = None
        self.sample_labels = None
        self.graph = None
        self.graph_kNN = None
        self.meld_op = None
        self.sample_densities = None
        self.estimates = {}

    def set_seed(self, seed):
        """Sets random seed.

        Parameters
        ----------
        seed : integer or numpy.RandomState
            Random state. Defaults to the global `numpy` random number generator

        Returns
        -------
        seed : integer or numpy.RandomState
            Newly set random seed.

        """

        self.seed = seed
        return self.seed

    def set_phate(self, data_phate):
        """Short summary.

        Parameters
        ----------
        data_phate : array, shape=[n_samples, 3]
            PHATE embedding for input data.

        Returns
        -------
        data_phate : array, shape=[n_samples, 3]
            Normalized PHATE embedding.

        """
        if not data_phate.shape[1] == 3:
            raise ValueError("data_phate must have 3 dimensions")
        if not np.isclose(data_phate.mean(), 0):
            # data_phate must be mean-centered
            data_phate = scipy.stats.zscore(data_phate, axis=0)
        self.data_phate = data_phate

    def fit_graph(self, data, **kwargs):
        """Fits a graphtools.Graph to input data

        Parameters
        ----------
        data : array, shape=[n_samples, n_observations]
            Input data
        **kwargs : dict
            Keyword arguments passed to gt.Graph()

        Returns
        -------
        graph : graphtools.Graph
            Graph fit to data

        """
        self.graph = gt.Graph(
            data, n_pca=100, use_pygsp=True, random_state=self.seed, **kwargs
        )
        return self.graph

    def fit_phate(self, data, **kwargs):
        """Generates a 3D phate embedding of input data

        Parameters
        ----------
        data : array, shape=[n_samples, n_observations]
            Description of parameter `data`.
        **kwargs : dict
            Keyword arguments passed to phate.PHATE().

        Returns
        -------
        data_phate : array, shape=[n_samples, 3]
            Normalized PHATE embedding for input data.

        """
        import phate

        self.set_phate(phate.PHATE(n_components=3, **kwargs).fit_transform(data))
        return self.data_phate

    def generate_ground_truth_pdf(self, data_phate=None):
        """Creates a random density function over input data.

        Takes a set of PHATE coordinates over a set of points and creates an underlying
        ground truth pdf over the points as a convex combination of the input phate
        coordinates.

        Parameters
        ----------
        data_phate : array, shape=[n_samples, 3]
            PHATE embedding for input data.

        Returns
        -------
        pdf
            Ground truth conditional probability of the sample given the data.

        """

        np.random.seed(self.seed)

        if data_phate is not None:
            self.set_phate(data_phate)
        elif self.data_phate is None:
            raise ValueError(
                "data_phate must be set prior to running generate_ground_truth_pdf()."
            )

        # Create an array of values that sums to 1
        data_simplex = np.sort(np.random.uniform(size=(2)))
        data_simplex = np.hstack([0, data_simplex, 1])
        data_simplex = np.diff(data_simplex)
        np.random.shuffle(data_simplex)

        # Weight each PHATE component by the simplex weights
        sort_axis = np.sum(self.data_phate * data_simplex, axis=1)

        # Pass the weighted components through a logit
        self.pdf = scipy.special.expit(sort_axis)
        return self.pdf

    def generate_sample_labels(self):
        np.random.seed(self.seed)

        # Create sample labels
        self.sample_indicator = np.random.binomial(1, self.pdf)
        self.sample_labels = np.array(
            ["ctrl" if ind == 0 else "expt" for ind in self.sample_indicator]
        )

    def calculate_MELD_likelihood(self, data=None, **kwargs):
        np.random.seed(self.seed)
        if not self.graph:
            if data is not None:
                self.fit_graph(data)
            else:
                raise NameError("Must pass `data` unless graph has already been fit")

        self.meld_op = meld.MELD(**kwargs, verbose=False).fit(self.graph)
        self.sample_densities = self.meld_op.transform(self.sample_labels)
        self.sample_likelihoods = meld.utils.normalize_densities(self.sample_densities)
        self.expt_likelihood = self.sample_likelihoods[
            "expt"
        ].values  # Only keep the expt condition
        return self.expt_likelihood

    def calculate_mse(self, estimate):
        """Calculated MSE between the ground truth PDF and an estimate"""
        return sklearn.metrics.mean_squared_error(self.pdf, estimate)
