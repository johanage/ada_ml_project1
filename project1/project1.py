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

from itertools import combinations_with_replacement
def make_design_matrix(xvec, p):
    xi = {"x%i"%i : xvec[i].ravel() for i in range(len(xvec))}
    keys = [key for key in xi.keys()]
    comb = []
    for p in range(1,p+1):
        comb += [x for x in combinations_with_replacement(keys, p )]
    #print(comb)
    X = np.ones((len(xi["x0"].ravel()), len(comb)+1))
    for j in range(len(X)):
        i = 0
        for c in comb:
            xij = 1
            for key in c:
                #print(key, i)
                xij *= xi[key][j]
            X[j,i+1] = xij
            i+=1
    return X

def ols_fp(xvec, f=FrankeFunction, p= 2, mu = 0, sigma = 1, return_betas=False):
    X = make_design_matrix(xvec = xvec, p = p)
    z = f(**{'x%i'%i: xvec[i].ravel() for i in range(len(xvec))})
    noise = np.random.normal(mu,sigma,size=z.shape)
    znoisy = z + noise
    znoisy_centered = znoisy - np.mean(znoisy)
    A = np.linalg.pinv(X.T@X)@X.T
    betahat = A@znoisy_centered
    znoisy_tilde = X@betahat + np.mean(znoisy)
    if return_betas:
        return znoisy_tilde, X, znoisy_centered, znoisy, betahat
    return znoisy_tilde, X, znoisy_centered, znoisy

from sklearn.model_selection import train_test_split
def ols_fp_train_test_split(X, y, **kwargs):
    ycentered = y - np.mean(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, ycentered, **kwargs)
    # computing beta params with train set
    A = np.linalg.pinv(Xtrain.T@Xtrain)@Xtrain.T
    betahat = A@ytrain
    ytilde_train = Xtrain@betahat + np.mean(y)
    ytilde_test = Xtest@betahat + np.mean(y)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest


def MSE(y, ytilde):
    return np.mean((y-ytilde)**2)

def Rscore(y, ytilde):
    mean = np.mean(y)
    SSres = np.sum((y-ytilde)**2)
    SStot = np.sum((y-mean)**2)
    return 1 - SSres/SStot
