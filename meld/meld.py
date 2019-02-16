import pygsp
from sklearn.base import BaseEstimator
import warnings

from . import utils


class MELD(BaseEstimator):
    """MELD operator for filtering signals over a graph.

    Parameters
    ----------
    beta : int
        Amount of smoothing to apply.
        Acts as 'p' parameter if fi == 'randomwalk'
    offset: int, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.
        Recommend using an eigenvalue from g based on the
        spectral distribution.
    order: int, optional, Default: 1
        Falloff and smoothness of the filter.
        High order leads to square-like filters.
    Solver : string, optional, Default: 'chebyshev'
        Method to solve convex problem.
        'chebyshev' uses a chebyshev polynomial approximation of the corresponding filter
        'exact' uses the eigenvalue solution to the problem
    M : int, optional, Default: 50
        Order of chebyshev approximation to use.
    lap_type : ('combinatorial', 'normalized'), Default: 'combinatorial'
        The kind of Laplacian to calculate
    suppress : bool
        Suppress warnings
    """

    def __init__(self, beta=1, offset=0, order=2, solver='chebyshev', M=50,
                 lap_type='combinatorial', suppress=False):

        self.suppress = suppress

        self.beta = beta
        self.offset = offset
        self.order = order
        self.solver = solver
        self.M = M
        self.lap_type = lap_type
        self.filt = None
        self.EES = None

    def fit(self, G):
        """Builds the MELD filter over a graph `G`.

        Parameters
        ----------
        G : graphtools.Graph object
            Graph to perform data smoothing over.

        """
        # Make sure that the input graph is a valid pygsp graph
        G = utils._check_pygsp_graph(G)

        if self.lap_type not in ('combinatorial', 'normalized'):
            raise TypeError("lap_type must be 'combinatorial'"
                            " or 'normalized'. Got: '{}'".format(self.lap_type))
        if G.lap_type != self.lap_type:
            warnings.warn(
                "Changing lap_type may require recomputing the Laplacian",
                RuntimeWarning)
            G.compute_laplacian(self.lap_type)

        # Generate MELD filter
        def filterfunc(x): return 1 / \
            (1 + (self.beta * x - self.offset)**self.order)

        G.estimate_lmax()
        self.filt = pygsp.filters.Filter(G, filterfunc)  # build filter

    def transform(self, RES, G):
        """Filters a signal `RES` over graph `G`.

        Parameters
        ----------
        RES : ndarray [n, p]
            2 dimensional input signal array to filter.

        G : graphtools.Graph object
            Graph to perform data smoothing over.

        Returns
        -------
        RES_nu : ndarray [n, p]
            Smoothed version of RES

        """
        G = utils._check_pygsp_graph(G)
        # Checking shape of RES and G match
        if RES.shape[0] != G.N:
            if len(RES.shape) > 1 and RES.shape[1] == G.N:
                warnings.warn(
                    "Input matrix is column-wise rather than row-wise. "
                    "transposing (output will be transposed)",
                    RuntimeWarning)
                RES = RES.T
            else:
                raise ValueError(
                    "Input data ({}) and input graph ({}) "
                    "are not of the same size".format(RES.shape, G.N))

        RES_nu = self.filt.filter(RES, method=self.solver,
                                order=self.M)  # apply filter

        return RES_nu

    def fit_transform(self, RES, G):
        """Builds the MELD filter over a graph `G` and filters a signal `RES`.

        Parameters
        ----------
        RES : ndarray [n, p]
            2 dimensional input signal array to filter.

        G : graphtools.Graph object
            Graph to perform data smoothing over.

        Returns
        -------
        EES : ndarray [n, p]
            Filtered version of RES

        """

        self.fit(G)
        self.EES = self.transform(RES, G)
        return self.EES
