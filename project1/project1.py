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
    X = np.ones((len(xi["x0"].ravel()), len(comb)+1))
    for j in range(len(X)):
        i = 0
        for c in comb:
            xij = 1
            for key in c:
                xij *= xi[key][j]
            X[j,i+1] = xij
            i+=1
    return X

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def ols_fp_wo_split(X, y, **kwargs):
    # computing beta params with train set
    A = np.linalg.pinv(X.T@X)@X.T
    betahat = A@y
    ytilde = X@betahat
    return ytilde, betahat

from sklearn.model_selection import train_test_split
def ols_fp_train_test_split(X, y, **kwargs):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, **kwargs)
    # computing beta params with train set
    ytilde_train, betahat = ols_fp_wo_split(X=Xtrain, y=ytrain-np.mean(ytrain))
    ytilde_train += np.mean(ytrain)
    ytilde_test = Xtest@betahat + np.mean(ytrain)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest

def scale_center_X(X):
    Xscaled = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    Xscaled[np.isnan(Xscaled)] = 0
    return Xscaled

def ridge_fp_wo_split(X, y, lmbda, centering = False, **kwargs):
    # centering inputs
    if centering == True:
        X = center_X(X=X)
    # computing beta params with train set
    p = X.shape[1]
    A = np.linalg.inv(X.T@X + np.eye(p, p)*lmbda)@X.T
    #print(np.eye(p,p)*lmbda)
    betahat = A@y
    ytilde = X@betahat
    return ytilde, betahat

def ridge_fp_train_test_split(X, y, lmbda, centering = False, **kwargs):
    """
    X has len=p not len=p+1 (cols with 1s is removed)

    """
    if np.sum(X[:,0] == np.ones(X.shape[1])) == X.shape[1]:
        raise ValueError(" First column of the design matrix needs to be removed")
    # centering inputs
    if centering == True:
        X = center_X(X=X)
    # split into training and test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, ycentered, **kwargs)
    # compute predictor for training and test set and add intercept
    ytilde_train, betahat = ridge_fp_wo_split(X=Xtrain, y=ytrain, lmbda=lmbda)
    ytilde_train = Xtrain@betahat + np.mean(y)
    ytilde_test = Xtest@betahat + np.mean(y)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest

from sklearn import linear_model
def lasso_fp_wo_split(X, y, lmbda, centering = False, **kwargs):
    # centering inputs
    if centering == True:
        X = center_X(X=X)
    clf = linear_model.Lasso(alpha = lmbda)
    clf.fit(X = X, y = y-np.mean(y))
    betahat = clf.coef_
    ytilde = X@betahat + np.mean(y)
    return ytilde, betahat

def lasso_fp_train_test_split(X, y, lmbda, centering = False, **kwargs):
    # centering inputs
    if centering == True:
        X = center_X(X=X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, **kwargs)
    clf = linear_model.Lasso(alpha = lmbda)
    clf.fit(X = Xtrain, y = ytrain)
    betahat = clf.coef_
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

def bias(y, ytilde):
    return np.mean((y-np.mean(ytilde))**2)

def resample(data):
    n = len(data.ravel())
    return data.ravel()[np.random.randint(0,n,n)].reshape(data.shape)

def bootstrap(data, k, keys_ops = {'mean' : np.mean, 'var' : np.var}):
    stats = {key : np.zeros(k) for key in keys_ops.keys()}
    for i in range(k):
        for key in stats.keys():
            data_resampled = resample(data=data)
            stats[key][i] = keys_ops[key](data_resampled)
    return stats

from sklearn import linear_model
def cross_validation(data, xvec, k, p, method, lmbda = None, scale_centering = False, scale = False, factor=1):
    """
    Args:
    data - data to be fitted, ndarray
    k - number of folds, int
    p - polynomial degree, int
    centering - centering the design matrix, bool 

    Out:
    mses - MSEs for each train test pair, np array with dim = (k,)

    """
    # b is flat array of data to fit
    b = data.ravel()
    # make array of indices of flattened data array
    indices = np.arange(len(b.ravel()))
    # make array of indices to split at
    isplit = np.full(k, len(b.ravel()) // k, dtype = int)
    isplit[:len(b.ravel()) % k] += 1
    isplit = np.cumsum(isplit)
    np.random.shuffle(indices)
    # split the datasets in k folds
    splits = np.split(np.arange(len(b.ravel()))[indices], isplit[:-1])
    # initiating the inferences
    mses_train = np.zeros((k))
    mses_test = np.zeros((k))
    bias_train = np.zeros((k))
    bias_test = np.zeros((k))
    var_train = np.zeros((k))
    var_test = np.zeros((k))
    for itest in range(k):
        # make sure that indices of flattened arraysare ints and not objects
        test = splits[itest]
        test = np.array(test,dtype=int)
        train = np.concatenate(np.delete(splits, itest, axis=0))
        train = np.array(train, dtype=int)
        if len(xvec) == 1:
            X = np.zeros((len(xvec[0]), p+1))
            X[:,0] = 1
            for deg in range(1,p+1):
                X[:,deg] = xvec[0]**deg
            Xtrain = X[train]
            Xtest = X[test]
        if len(xvec) == 2:
            #Xtrain = make_design_matrix(xvec = np.array([x.ravel()[train] for x in xvec]), p = p)
            Xtrain = create_X(x = xvec[0].ravel()[train], y = xvec[1].ravel()[train], n = p)
            #Xtest = make_design_matrix(xvec = np.array([x.ravel()[test] for x in xvec]), p = p)
            Xtest = create_X(x = xvec[0].ravel()[test], y = xvec[1].ravel()[test], n = p)
        if method == "ols":
            data_tilde_train, betahat = ols_fp_wo_split(X = Xtrain, y = b[train])
            data_tilde_test = Xtest@betahat
        if method == "lasso":
            Xtrain = Xtrain[:,1:]
            if scale_centering == True:
                mean_Xtrain = np.mean(Xtrain, axis=0)
                std_Xtrain = np.std(Xtrain, axis = 0)
                Xtest =  (Xtest[:,1:] - mean_Xtrain)/std_Xtrain
                Xtrain = scale_center_X(X=Xtrain)
            if lmbda is None:
                raise ValueError("Lambda value has not been set")
            clf_train = linear_model.Lasso(alpha = lmbda)
            clf_train.fit(X=Xtrain, y = b[train]-np.mean(b[train]))
            betahat = clf_train.coef_
            data_tilde_train = Xtrain@betahat + np.mean(b[train])
            data_tilde_test = Xtest@betahat + np.mean(b[train])
        if method == "ridge":
            Xtrain = Xtrain[:,1:]
            if scale_centering == True:
                mean_Xtrain = np.mean(Xtrain, axis=0)
                std_Xtrain = np.std(Xtrain,axis=0)
                Xtest =  (Xtest[:,1:]-mean_Xtrain)/std_Xtrain  
                Xtrain = scale_center_X(X=Xtrain)
            if lmbda is None:
                raise ValueError("Lambda value has not been set")
            data_tilde_train, betahat = ridge_fp_wo_split(X=Xtrain, y=b[train]-np.mean(b[train]), lmbda=lmbda)
            data_tilde_train = (data_tilde_train + np.mean(b[train]))
            data_tilde_test = Xtest@betahat + np.mean(b[train])
            data_tilde_test = Xtest@betahat + np.mean(b[train])
        mses_train[itest] = MSE(y = b[train], ytilde = data_tilde_train)
        mses_test[itest]  = MSE(y = b[test], ytilde = data_tilde_test)
        bias_train[itest] = bias(y = b[train], ytilde = data_tilde_train)
        bias_test[itest]  = bias(y = b[test], ytilde = data_tilde_test)
        var_train[itest]  = np.var(data_tilde_train)
        var_test[itest]   = np.var(data_tilde_test)

    return mses_train, mses_test, bias_train, bias_test, var_train, var_test

