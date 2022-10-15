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
    # store all independent variables in a dict
    xi = {"x%i"%i : xvec[i].ravel() for i in range(len(xvec))}
    # make all possible combinations of the indep var with replacement
    keys = [key for key in xi.keys()]
    comb = []
    for p in range(1,p+1):
        comb += [x for x in combinations_with_replacement(keys, p )]
    # add the values to the possible combinations of indep variables up to power p
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
    # flatten array of shape is larger than 1
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))
    # make the designmatrix if two independent variables
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def ols_fp_wo_split(X, y, **kwargs):
    # computing beta params with train set
    A = np.linalg.pinv(X.T@X)@X.T
    # compute the coef
    betahat = A@y
    # compute prediction
    ytilde = X@betahat
    return ytilde, betahat

from sklearn.model_selection import train_test_split
def ols_fp_train_test_split(X, y, **kwargs):
    # split in train test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, **kwargs)
    # computing beta params and prediction with train set
    ytilde_train, betahat = ols_fp_wo_split(X=Xtrain, y=ytrain-np.mean(ytrain))
    ytilde_train += np.mean(ytrain)
    ytilde_test = Xtest@betahat + np.mean(ytrain)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest

from sklearn.preprocessing import StandardScaler
def scale_center_X(X, test = False):
    """
    Scales and centers the design matrix
    Args:
    X - design matrix, ndarray
    test - if True test the function, bool
    Out:
    Xscaled - scaled and centered design matrix, ndarray 
    """
    Xscaled = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    Xscaled[np.isnan(Xscaled)] = 0
    if test:
        Xscaled_SL = scaler.transform(X)
        if not np.sum(Xscaled_SL == Xscaled) == len(Xscaled.ravel()):
            raise ValueError("The scaling and centering does not give the same results as the SL StandardScaler")
    return Xscaled

def ridge_fp_wo_split(X, y, lmbda, **kwargs):
    """
    Perform ridge regression without splitting the data into train test sets.
    Args:
    X - desing matrix, ndarray
    y - data, ndarray
    lmbda - regularization param, float
    kwargs - key word arguments for the regression method
    Out:
    ytilde - predicted data, ndarray
    betahat - coefficients, ndarray
    """
    # computing beta params with train set
    p = X.shape[1]
    # scale and center the design matrix
    X_scaled = scale_center_X(X = X)
    A = np.linalg.inv(X_scaled.T@X_scaled + np.eye(p, p)*lmbda)@X_scaled.T
    # predict on centered data
    ymean  = np.mean(y)
    # compute coef
    betahat = A@(y-ymean)
    # compute prediction
    ytilde = X_scaled@betahat + ymean
    return ytilde, betahat

def ridge_fp_train_test_split(X, y, lmbda, **kwargs):
    """
    Performs Ridge regression taking in a dataset, a designmatrix and a regularization parameter lambda.
    X has len=p not len=p+1 (cols with 1s is removed)
    Args:
    X - desing matrix, ndarray
    y - data, ndarray
    lmbda - regularization param, float
    kwargs - key word arguments for the regression method
    Out:
    ytilde - predicted data, ndarray
    betahat - coefficients, ndarray
    """
    if np.sum(X[:,0] == np.ones(X.shape[1])) == X.shape[1]:
        raise ValueError(" First column of the design matrix needs to be removed")
    # split into training and test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, **kwargs)
    # compute predictor for training and test set and add intercept
    ytilde_train, betahat = ridge_fp_wo_split(X=Xtrain, y=ytrain - np.mean(ytrain), lmbda=lmbda)
    ytilde_train = Xtrain@betahat + np.mean(ytrain)
    ytilde_test = Xtest@betahat + np.mean(ytrain)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest

from sklearn import linear_model
def lasso_fp_wo_split(X, y, lmbda, **kwargs):
    """
    Performs Lasso regression taking in a dataset, a designmatrix and a regularization parameter lambda.
    Args:
    X - desing matrix, ndarray
    y - data, ndarray
    lmbda - regularization param, float
    kwargs - key word arguments for the regression method
    Out:
    ytilde - predicted data, ndarray
    betahat - coefficients, ndarray
    """
    clf = linear_model.Lasso(alpha = lmbda, **kwargs)
    X_scaled = scale_center_X(X = X)
    clf.fit(X = X_scaled, y = y-np.mean(y))
    betahat = clf.coef_
    ytilde = X_scaled@betahat + np.mean(y)
    return ytilde, betahat

def lasso_fp_train_test_split(X, y, lmbda, **kwargs):
    """
    Performs Lasso regression taking in a dataset, a designmatrix and a regularization parameter lambda.
    Args:
    X - desing matrix, ndarray
    y - data, ndarray
    lmbda - regularization param, float
    kwargs - key word arguments for the regression method
    Out:
    y{train,test} - training and test set
    X{train,test} - training and test design matrix
    ytilde_{test, train} - predicted data for test and train sets, ndarray
    betahat - coefficients, ndarray
    """

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, **kwargs)
    clf = linear_model.Lasso(alpha = lmbda, fit_intercept=True)
    clf.fit(X = Xtrain, y = ytrain-np.mean(ytrain))
    betahat = clf.coef_
    ytilde_train = Xtrain@betahat + np.mean(ytrain)
    ytilde_test = Xtest@betahat + np.mean(ytrain)
    return ytilde_train, ytilde_test, betahat, Xtrain, Xtest, ytrain,ytest


def MSE(y, ytilde):
    """
    Computes the MSE of predicted data ytilde with respect to the true data y.
    Both for 1D and 2D arrays.
    """
    if len(ytilde.shape) == 1:
        return np.mean((y - ytilde)**2)
    if len(ytilde.shape) == 2:
        # average over resamples first
        return np.mean( np.mean((y[:,np.newaxis]-ytilde)**2, axis=1))

def Rscore(y, ytilde):
    """
    Computes the R2 score for the predicted data ytilde given the true data y.
    """
    mean = np.mean(y)
    SSres = np.sum((y-ytilde)**2)
    SStot = np.sum((y-mean)**2)
    return 1 - SSres/SStot

def exp_bias(y, exp_ytilde):
    # compute expectation of square bias
    return np.mean((y-exp_ytilde)**2)

def resample(data, design_matrix):
    # resample drawing indices at random with replacement
    n = len(data.ravel())
    indices = np.random.randint(0,n,n)
    data_resampled = data.ravel()[indices].reshape(data.shape)
    design_matrix_resampled = design_matrix[indices]
    return data_resampled, design_matrix_resampled

def bootstrap(data, k, keys_ops = {'mean' : np.mean, 'var' : np.var}):
    """
    Perforn bootstrap inference given by key_ops on data k times.
    """
    stats = {key : np.zeros(k) for key in keys_ops.keys()}
    for i in range(k):
        for key in stats.keys():
            data_resampled = resample(data=data)
            stats[key][i] = keys_ops[key](data_resampled)
    return stats

from sklearn import linear_model
def kfold(data, k, random_state = 42):
    """
    Compute indices for the kfold split.
    Args: data to fit, ndarray
    Out: splits, array of indices on where to split the data array
    """
    np.random.seed(random_state)
    # b is flat array of data to fit
    b = data.ravel()
    # make array of indices of flattened data array
    indices = np.arange(len(b.ravel()))
    # make array of indices to split at
    isplit = np.full(k, len(b.ravel()) // k, dtype = int)
    # if equal split cannot be made then add 1 data point to the len(b.ravel()) % k first folds
    isplit[:len(b.ravel()) % k] += 1
    # cumsum for index to split at
    isplit = np.cumsum(isplit)
    # shuffle indices to get random data points in folds
    np.random.shuffle(indices)
    # split the datasets in k folds
    splits = np.split(np.arange(len(b.ravel()))[indices], isplit[:-1])
    return splits

def cross_validation(data, splits, xvec, k, p, method, lmbda = None, scale_centering = True):
    """
    Args:
    data - data to be fitted, ndarray
    k - number of folds, int
    p - polynomial degree, int
    centering - centering the design matrix, bool 

    Out:
    mses - MSE train test pair, float

    """
    b = data.ravel()
    data_train = np.ones((len(data.ravel()) - min([len(x) for x in splits]), k))*np.nan
    data_test = np.ones((max([len(x) for x in splits]), k))*np.nan
    data_tilde_train = np.ones((len(data.ravel()) - min([len(x) for x in splits]), k))*np.nan
    data_tilde_test = np.ones((max([len(x) for x in splits]), k))*np.nan
    #data_tilde_train = np.zeros((len(data.ravel()-len(splits)[0]
    for itest in range(k):
        # make sure that indices of flattened arraysare ints and not objects
        test = splits[itest]
        test = np.array(test,dtype=int)
        train = np.concatenate(np.delete(splits, itest, axis=0))
        train = np.sort(np.array(train, dtype=int))
        data_train[:len(train),itest] = b[train]
        data_test[:len(test),itest] = b[test]
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
            meanXtrain =  np.mean(Xtrain,axis=0)
            if scale_centering:
                Xtrain_c = Xtrain[:,1:]/np.std(Xtrain[:,1:],axis=0)
                Xtest_c = Xtest[:,1:]/np.std(Xtrain[:,1:],axis=0)
            else:
                Xtrain_c = Xtrain - meanXtrain
                Xtest_c = Xtest - meanXtrain
            data_tilde_train[:len(train),itest], betahat = ols_fp_wo_split(X = Xtrain, y = b[train]-np.mean(b[train]))
            data_tilde_test[:len(test),itest] = Xtest@betahat + np.mean(b[train])
        if method == "lasso":
            Xtrain = Xtrain[:,1:]
            Xtest = Xtest[:,1:]
            if scale_centering == True:
                mean_Xtrain = np.mean(Xtrain, axis=0)
                std_Xtrain = np.std(Xtrain, axis = 0)
                Xtest =  (Xtest - mean_Xtrain)/std_Xtrain
                Xtrain = scale_center_X(X=Xtrain)
            if lmbda is None:
                raise ValueError("Lambda value has not been set")
            clf_train = linear_model.Lasso(alpha = lmbda, fit_intercept=True)
            clf_train.fit(X=Xtrain, y = b[train]-np.mean(b[train]))
            betahat = clf_train.coef_
            data_tilde_train[:len(train),itest] = Xtrain@betahat + np.mean(b[train])
            data_tilde_test[:len(test),itest] = Xtest@betahat + np.mean(b[train])
        if method == "ridge":
            Xtrain = Xtrain[:,1:]
            Xtest = Xtest[:,1:]
            if scale_centering == True:
                mean_Xtrain = np.mean(Xtrain, axis=0)
                std_Xtrain = np.std(Xtrain,axis=0)
                Xtest =  (Xtest - mean_Xtrain)/std_Xtrain  
                Xtrain = scale_center_X(X=Xtrain)
            if lmbda is None:
                raise ValueError("Lambda value has not been set")
            data_tilde_train[:len(train),itest], betahat = ridge_fp_wo_split(X=Xtrain, y=b[train]-np.mean(b[train]), lmbda=lmbda)
            data_tilde_train[:len(train),itest] += np.mean(b[train])
            data_tilde_test[:len(test),itest] = Xtest@betahat + np.mean(b[train])

    return data_train, data_test, data_tilde_train, data_tilde_test

# IGNORE
# if y = \beta is assumed
def beta_lasso(y, betahat, lmbda):
    """
    Args:
    beta - lasso params
    y - data to fit
    Out: update beta params
                    yi - lambda/2, if yi > lambda/2
    beta_lasso =    yi + lambda/2, if yi < -lambda/2
                    0            ,if |yi|<=lambda/2
    """
    beta_lasso = beta.copy()*np.nan
    inds_above_lhalf = y > lmbda/2
    inds_below_minlhalf = y < -lmbda/2
    inds_other = np.abs(y) <= lmbda/2
    beta_lasso[inds_above_lhalf] = y[inds_above_lhalf]-lmbda/2
    beta_lasso[inds_below_lhalf] = y[inds_below_lhalf]+lmbda/2
    beta_lasso[inds_other] = 0
    return beta_lasso

def Cmin_lasso(y, betahat, lmbda):
    return -2*np.sum(y-betahat) + lmbda*np.sum(np.sign(betahat))


def OWN_lasso_fp_wo_split(X, y, lmbda, **kwargs):
    threshold = kwargs['threshold']
    maxiter = kwargs['maxiter']
    # initating betas with OLS and then improving betas by lasso
    ytilde, betahat = ols_fp_wo_split(X = X, y = y)
    for i in maxiter:
        betahat = beta_lasso(y = y, betahat = betahat, lmbda = lmbda)
        if Cmin_lasso(y = y, betahat = betahat, lmbda = lmbda) < threshold:
            ytilde = X @ betahat
            return ytilde, betahat
