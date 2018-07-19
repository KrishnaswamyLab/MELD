import numpy as np
import pygsp
import graphtools
import scipy.sparse as sparse


def meld(X, gamma, g, solver='cheby', fi='regularizedlaplacian', alpha=2):
    """
    Performs convex meld on the input signal.
    This function solves:

        (1) :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \gamma \| \nabla x\|_2^2`

        OR
        regularizes (1) using \inner{X, randomwalk(L)*X}, the p-step random walk (math to come)

    Note the nice following relationship for (1):

        (2) :math:`x^T L x = \| \nabla x\|_2^2`

    Also note that the solution to (1) may be phrased as the lowpass filter:

        (3) :math:`sol = h(L)x` with :math:`h(\lambda) := \frac{1}{1+\gamma\lambda}`

    We use (3) by default as it is faster in the case of few input signals.


    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input signal array to meld.
    gamma : int
        Amount of smoothing to apply.  Acts as 'p' parameter if fi == 'randomwalk'
    g : graphtools.Graph object
        Graph to perform data smoothing over.
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

    if not (isinstance(g,  graphtools.base.BaseGraph) or
            isinstance(g, pygsp.graphs.Graph)):
        raise TypeError("Input graph should be of type graphtools.BaseGraph")

    if X.shape[0] != g.N:
        if X.shape[1] == g.N:
            print(
                "input matrix is column-wise rather than row-wise. "
                "transposing (output will be transposed)")
            X = X.T
        else:
            raise ValueError(
                "Input data and input graph are not of the same size")

    if fi == 'randomwalk':
        # used for random walk stochasticity
        D = sparse.diags(np.ravel(np.power(g.W.sum(0), -1)), 0).tocsc()
    if solver == 'matrix':
        # use matrix inversion / powering
        I = sparse.identity(g.N)
        if fi == 'regularizedlaplacian':  # fTLf
            mat = sparse.linalg.inv((I + gamma * g.L))

        elif fi == 'randomwalk':  # p-step random walk
            mat = (alpha * I - (g.L * D))**gamma

        sol = mat.T @ X  # apply the matrix
        sol = np.squeeze(np.asarray(sol))  # deliver a vector

    else:
        # use approximations
        if fi == 'regularizedlaplacian':  # fTLf
            filterfunc = lambda x: 1 / (1 + gamma * x)

        elif fi == 'randomwalk':  # p-step random walk
            L_bak = g.L
            # change the eigenbasis by normalizing by degree (stochasticity)
            g.L = (L_bak * D).T
            filterfunc = lambda x: (alpha - x)**gamma

        filt = pygsp.filters.Filter(g, filterfunc)  # build filter
        sol = filt.filter(X)  # apply filter
        if fi == 'randomwalk':
            g.L = L_bak  # restore L

    return sol
