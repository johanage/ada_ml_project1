import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
def FrankeFunction(**kwargs):
    x = kwargs['x0']
    y = kwargs['x1']
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def plot_surface(ax = None, fig = None, **kwargs):
    z = FrankeFunction(**kwargs)
    # Plot the surface.
    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig, ax

def make_design_matrix(xvec, deg):
    X = np.ones(tuple([xvec.shape[0], deg+1] + list(xvec.shape[1:])))*np.nan
    for i in range(X.shape[0]):
        for n in range(deg+1):
            if n == 0:
                X[i, n] = 2
            else:
                X[i, n] = xvec[i]**n
    ds = xr.Dataset(coords = {'x%i'%i : (['nx%i'%i for i in range(X.shape[0])], X[i,1]) for i in range(X.shape[0])},
                    data_vars = {'design_matrix' : (['vars', 'deg'] + ['nx%i'%i for i in range(X.shape[0])], X)})
    return X.T, ds

def ols_fp(xvec, f=FrankeFunction, deg = 2):
    X = make_design_matrix(xvec = xvec, deg = deg)
    z = f(**{'x%i'%(i) : xvec[i] for i in range(len(xvec))})
    noise = np.random.normal(0,1, size=z.shape)
    znoisy = z + noise
    betahat = np.array([np.linalg.inv(X[i].T@X[i])@X[i].T@znoisy for i in range(len(X))])
    znoisy_tilde = np.sum(np.array([X[i]@betahat[i] for i in range(len(X))]), axis=0)
    return znoisy_tilde, X, noise
