# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.metrics import *
from scipy.optimize import minimize

def obj_function(w,Mi,ai,regcoef1,regcoef2):
    J = -1*np.dot(w.T, ai) + regcoef1*np.dot(np.dot(w.T, Mi), w) + regcoef2*(np.linalg.norm(w, ord=2, keepdims=True))**2
    return J

def con():
    cons = ({'type': 'eq', 'fun': lambda w:sum(w)-1}, {'type': 'ineq', 'fun': lambda w:w})
    return cons

def optimize_weights(x0, fun):
    res = minimize(fun, x0, method='SLSQP')
    return res.x

def hsic_kernel_weights_norm(Kernels_list, adjmat,dim, regcoef1, regcoef2):
    adjmat = np.array(adjmat)

    num_kernels = np.size(Kernels_list, 0)
    weight_v = np.zeros((num_kernels, 1))
    y = adjmat

    # Graph based kernel
    if dim == 1:
        ideal_kernel = np.dot(y, y.T)
    else:
        ideal_kernel = np.dot(y.T, y)

    # ideal_kernel=Knormalized(ideal_kernel)
    # print(type(ideal_kernel))
    N_U = np.size(ideal_kernel, 0)
    l = np.ones((N_U, 1))
    H = np.eye(N_U, dtype=float) - np.dot(l, l.T)/N_U # H:NxN的矩阵

    M = np.zeros((num_kernels, num_kernels))
    for i in range(num_kernels):
        for j in range(num_kernels):
            kk1 = np.dot(np.dot(H, Kernels_list[i, :, :]), H)
            kk2 = np.dot(np.dot(H, Kernels_list[j, :, :]), H)
            mm = np.trace(np.dot(kk1.transpose(), kk2))
            m1 = np.trace(np.dot(kk1, kk1.transpose()))
            m2 = np.trace(np.dot(kk2, kk2.transpose()))
            M[i, j] = mm / (np.sqrt(m1) * np.sqrt(m2))
    d_1 = sum(M)
    D_1 = np.diag(d_1)
    LapM = D_1 - M

    a = np.zeros((num_kernels, 1))
    for i in range(num_kernels):
        kk = np.dot(np.dot(H, Kernels_list[i, :, :]), H)
        aa = np.trace(np.dot(kk.transpose(), ideal_kernel))
        a[i] = aa*((N_U-1)**-2)

    v = np.random.rand(num_kernels, 1)
    falpha = lambda v:obj_function(v, LapM, a, regcoef1, regcoef2)
    x_alpha = optimize_weights(v, falpha)
    weight_v = x_alpha
    return weight_v

def line_map(X):
    col_X = np.size(X, 1)
    Mapping_X = []
    for i in range(col_X):
        col_v = (X[:, i] - X[:, i].min()) / float(X[:, i].max() - X[:, i].min())
        Mapping_X.append(col_v)
    Mapping_X = np.array(Mapping_X)
    Mapping_X = Mapping_X.T
    return Mapping_X

def combine_kernels(weights, kernels):
    result = np.zeros(kernels[1, :, :].shape)
    n = len(weights)
    for i in range(n):
        result = result + weights[i] * kernels[i, :, :]
    return result

def kernel_RBF(X, Y, gamma):
    r2 = np.tile(np.sum(X**2, 1), (np.size(Y, 0),1)).T + np.tile(np.sum(Y**2, 1), (np.size(X, 0), 1)) - 2 * np.dot(X, Y.T)
    k = np.exp(-r2 * gamma)
    return k

def skl_mk_svm(train_x,feature_id,train_y,test_x,test_y,c,gamma_list):
    predict_y = []
    Scores = []
    kernel_weights = []
    m = int(len(feature_id) / 2)
    num_train_samples = np.size(train_x, 0)
    num_test_samples = np.size(test_x, 0)
    #1.computer training and test kernels (with RBF)
    K_train = []
    K_test = []
    for i in range(m):
        kk_train = kernel_RBF(train_x[:, feature_id[2*i]:feature_id[2*i+1]], train_x[:, feature_id[2*i]:feature_id[2*i+1]], gamma_list[i])
        K_train.append(kk_train)

        kk_test = kernel_RBF(test_x[:, feature_id[2*i]:feature_id[2*i+1]], train_x[:, feature_id[2*i]:feature_id[2*i+1]], gamma_list[i])
        K_test.append(kk_test)

    K_train = np.array(K_train) #这时K_train是三维矩阵了
    K_test = np.array(K_test)  #这时K_test是三维矩阵了

    # 2.multiple kernel learning
    kernel_weights = hsic_kernel_weights_norm(K_train, train_y, 1, 0.1, 0.01)
    # kernel_weights = np.ones((m, 1))
    # kernel_weights = kernel_weights / m
    kernel_weights = kernel_weights.reshape(m,).tolist()
    print(kernel_weights)

    K_train_com = combine_kernels(kernel_weights, K_train)
    K_test_com = combine_kernels(kernel_weights, K_test)

    K_train_com.tolist()
    K_test_com.tolist()
    # 3.train and test model
    # K_train_com = np.insert(K_train_com, 0, [j for j in range(1, num_train_samples+1)], axis=1)
    # cg_str = ['-t 4 -c '+ str(c)+' -b 1 -q']
    train_y = train_y.reshape(len(K_train_com),).tolist()
    clf = svm.SVC(C=c,kernel='precomputed')
    clf.fit(K_train_com, train_y)
    # K_test_com = np.insert(K_test_com, 0, [j for j in range(1, num_test_samples+1)], axis=1)
    y_pred = clf.predict(K_test_com)
    precision = precision_score(test_y, y_pred, average="weighted")
    return y_pred, precision