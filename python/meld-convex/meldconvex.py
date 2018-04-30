import numpy as np
import pygsp
import graphtools

def meld(X, gamma, g, method = 'lsq'):
    """
    Performs convex meld on the input signal. 
    This function solves:

        (1) :math:`sol = argmin_{z} \frac{1}{2}\|x - z\|_2^2 + \gamma \| \nabla x\|_2^2`

    Note the nice following relationship:

        (2) :math:`x^T L x = \| \nabla x\|_2^2`

    Also note that the solution to (1) may be phrased as the lowpass filter:

        (3) :math:`sol = h(L)x` with :math:`h(\lambda) := \frac{1}{1+\gamma\lambda}`
        

    Parameters
    ----------
    X : ndarray [n, p]
        2 dimensional input signal array to meld.

    gamma : int
        Amount of smoothing to apply.

    g : graphtools.Graph object
        Graph to perform data smoothing over.

    method : string, optional, Default: 'lsq'
        Method to solve convex problem. 'lsq' is the only implemented version currently.
        
    Returns
    -------
    sol : ndarray [n, p]
        2 dimensional array of smoothed input signals
    """
    if not isinstance(method, str):
        raise TypeError("Input method should be a string")
    method = method.lower()
    if method != 'lsq':
        raise NotImplementedError('{} method is not currently implemented.'.format(method))
    
    if not (isinstance(g, graphtools.BaseGraph) or isinstance(g,pygsp.graphs.Graph)):
        raise TypeError("Input graph should be of type graphtools.BaseGraph")
    if X.shape[0] != g.N:
        if X.shape[1] == g.N:
            print("input matrix is column-wise rather than row-wise. transposing (output will be transposed)")
            X = X.T;
        else:
            raise ValueError("Input data and input graph are not of the same size")
    
    mat = (np.eye(g.N) + gamma * g.L)
    sol = np.linalg.lstsq(mat, X)
    
    return sol
            
    
