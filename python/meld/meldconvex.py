import numpy as np
import pygsp
import graphtools
import graphtools.base
import scipy.sparse as sparse
import inspect
from sklearn.cluster import KMeans
from . import utils
from sklearn import preprocessing


def meld(X, G, beta, offset = 0, order = 1, solver='cheby', fi='regularizedlaplacian', alpha=2):
    """
    Performs convex meld on the input signal.
    This function solves:

        (1) :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \beta \| \nabla x\|_2^2`

        OR
        regularizes (1) using \inner{X, randomwalk(L)*X}, the p-step random walk (math to come)

    Note the nice following relationship for (1):

        (2) :math:`x^T L x = \| \nabla x\|_2^2`

    Also note that the solution to (1) may be phrased as the lowpass filter:

        (3) :math:`sol = h(L)x` with :math:`h(\lambda) := \frac{1}{1+\beta\lambda}`

    We use (3) by default as it is faster in the case of few input signals.


    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input signal array to meld.
    G : graphtools.Graph object
        Graph to perform data smoothing over.
    beta : int
        Amount of smoothing to apply.  Acts as 'p' parameter if fi == 'randomwalk'
    offset: int, optional, Default: 0
        Amount to shift the MELD filter in the eigenvalue spectrum.  Recommend using
        an eigenvalue from g based on the spectral distribution.
    order: int, optional, Default: 1
        Falloff and smoothness of the filter.   High order leads to square-like filters.
    Solver : string, optional, Default: 'cheby'
        Method to solve convex problem.
        'cheby' uses a chebyshev polynomial approximation of the corresponding filter
        'matrix' solves the convex problem exactly
    fi: string, optional, Default: 'regularizedlaplacian'
        Filter to use for (1).
        'regularizedlaplacian' is the exact solution of (1)
        'randomwalk' is a randomwalk polynomial that is related to diffusion via rw = ((alpha-1)I+P)^t

    Returns
    -------
    sol : ndarray [n, p]
        2 dimensional array of smoothed input signals
    """

    if not isinstance(solver, str):
        raise TypeError("Input method should be a string")
    solver = solver.lower()
    if solver not in ['matrix', 'cheby']:
        raise NotImplementedError(
            '{} solver is not currently implemented.'.format(solver))

    if not isinstance(fi, str):
        raise TypeError("Input filter should be a string")
    fi = fi.lower()
    if fi not in ['regularizedlaplacian', 'randomwalk']:
        raise NotImplementedError(
            '{} filter is not currently implemented.'.format(fi))

    if not isinstance(G, pygsp.graphs.Graph):
        if isinstance(G, graphtools.base.BaseGraph):
            raise TypeError(
                "Input graph should be of type pygsp.graphs.Graph. "
                "When using graphtools, use the `use_pygsp=True` flag.")
        else:
            raise TypeError(
                "Input graph should be of type pygsp.graphs.Graph. "
                "Got {}".format(type(G)))

    if X.shape[0] != G.N:
        if len(X.shape) > 1 and X.shape[1] == G.N:
            print(
                "input matrix is column-wise rather than row-wise. "
                "transposing (output will be transposed)")
            X = X.T
        else:
            raise ValueError(
                "Input data ({}) and input graph ({}) "
                "are not of the same size".format(X.shape, G.N))

    if fi == 'randomwalk':
        # used for random walk stochasticity
        D = sparse.diags(np.ravel(np.power(G.W.sum(0), -1)), 0).tocsc()
    if solver == 'matrix':
        # use matrix inversion / powering
        I = sparse.identity(G.N)
        if fi == 'regularizedlaplacian':  # fTLf
            mat = sparse.linalg.inv((I + np.matrix_power(beta * G.L-offset*I, order)).tocsc())
        elif fi == 'randomwalk':  # p-step random walk
            mat = (alpha * I - (G.L * D))**beta

        sol = mat.T @ X  # apply the matrix
        sol = np.squeeze(np.asarray(sol))  # deliver a vector

    else:
        # use approximations
        if fi == 'regularizedlaplacian':  # fTLf
            filterfunc = lambda x: 1 / (1 + (beta * x-offset)**order)

        elif fi == 'randomwalk':  # p-step random walk
            L_bak = G.L
            # change the eigenbasis by normalizing by degree (stochasticity)
            G.L = (L_bak * D).T
            filterfunc = lambda x: (alpha - x)**beta

        G.estimate_lmax()
        filt = pygsp.filters.Filter(G, filterfunc)  # build filter
        sol = filt.filter(X)  # apply filter
        if fi == 'randomwalk':
            G.L = L_bak  # restore L

    sol = utils.convert_to_same_format(sol, X)

    return sol



def spectrogram_clustering(G, s = None,  t = 10, saturation = 0.5, use_diffop = True, kernel = None, clusterobj = None, nclusts = 5, precomputed_nwgft = None, **kwargs):
    saturation_func = lambda x,alpha: np.tanh(alpha * np.abs(x.T)) #TODO: extend to allow different saturation functions
    if not(isinstance(clusterobj, KMeans)):
        #todo: add support for other clustering algorithms
        if clusterobj is None:
            clusterobj = KMeans(n_clusters = nclusts, **kwargs)
        else:
            raise TypeError(
                "Currently only sklearn.cluster.KMeans is supported for clustering object. "
                "Got {}".format(type(clusterobj)))

    if precomputed_nwgft is not None: #we don't need to do much if we have a precomputed nwgft
        C = precomputed_nwgft
    else:
        #check that signal and graph are defined
        if s is None:
            raise RuntimeError(
                    "If no precomputed_nwgft, then a signal s should be supplied.")
        if not isinstance(G, pygsp.graphs.Graph):
            if isinstance(G, graphtools.base.BaseGraph):
                raise TypeError(
                    "Input graph should be of type pygsp.graphs.Graph. "
                    "When using graphtools, use the `use_pygsp=True` flag.")
            else:
                raise TypeError(
                    "Input graph should be of type pygsp.graphs.Graph. "
                    "Got {}".format(type(G)))
        #build kernel

        if use_diffop is False:
            if kernel and not(inspect.isfunction(kernel)):
                    raise TypeError(
                        "Input kernel should be a lambda function (accepting "
                        "eigenvalues of the graph laplacian) or none. "
                        "Got {}".format(type(kernel)))
            if kernel is None:
                kernel = lambda x:  np.exp((-t*x)/G.lmax) #definition of the heat kernel

            ke = kernel(G.e) #eval kernel over eigenvalues of G
            ktrans = np.sqrt(G.N) * (G.U @ np.multiply(ke[:,None],G.U.T)) #vertex domain translation of the kernel.

            C = np.empty((G.N,G.N))

            for i in range(0,G.N): #build frame matrix
                kmod = np.matlib.repmat(ktrans[:,i], 1,G.N) # copy one translate Ntimes
                kmod = np.reshape(kmod, (G.N, G.N)).T
                kmod = (G.U/G.U[:,0]) * kmod # modulate the copy at each frequency of G
                kmod = kmod / np.linalg.norm(kmod, axis = 0) #normalize it
                C[:,i] = kmod.T@s # compute nwgft frame
        else:
            window = preprocessing.normalize(np.linalg.matrix_power(G.diff_op.toarray(), t), 'l2', axis=0).T
            C = np.multiply(window,s[:,None])
            C = G.gft(C)

    labels = clusterobj.fit_predict(saturation_func(C,saturation))

    return C, labels, saturation_func
