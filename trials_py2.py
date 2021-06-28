


import numpy as np
from matplotlib import pyplot as plt
from julia.api import Julia
jl = Julia(compiled_modules=False)
from sys import path
from dpmmpython.priors import niw
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.dpmmwrapper import DPMModel
import random

#path[4] = '/Users/irita/Documents/Master/Research/PhD_research/BNP/Code - Or/DPMM_python_my_wrapper/dpmmpython' 
#import dpmmpython
#print(dpmmpython)
#1/0


random.seed(10)

# --- Toy data #1:
# N = 20000 #Number of points
# D = 2 # Dimension
# modes = 20 # Number of Clusters
# var_scale = 100.0 # The variance of the MV-Normal distribution where the clusters means are sampled from.

# --- Toy data #2:
N = 100   # 10000 # number of points
D = 2   # data dimension
modes = 3   # number of modes
var_scale = 80.0

# --- Extract the data in the shape (D,N).
data, labels = DPMMPython.generate_gaussian_data(N, D, modes, var_scale)
print(labels)
# Changing the lables to be incorrect (to see how the splits work)
#labels[labels==3] = 2
#labels[labels==4] = 3
#labels[labels==5] = 4
#labels[labels==6] = 130

# --- hyper params #1:
# hyper_prior = niw(1,np.zeros(D),5,np.eye(D)*0.5)
# alpha = 10.
# iters = 500

# --- hyper params #2:
init_clusters = np.unique(labels).size
m = np.zeros(D)
k = init_clusters  #1.0
nu = 130.  # should be > D
psi = np.cov(data)*0.01  # shape (D,D)
hyper_prior = niw(k, m, nu, psi)
alpha = 1.
iters = 50 #200

# --- Print original label counts: (need to fix, see how it's done in trials.jl)
# label_counts = np.zeros(init_clusters)
# for i in range(len(labels)):
#     l = int(labels[i]-1)
#     label_counts[l] = label_counts[l] + 1

# for i in range(len(label_counts)):
#     print("label ", str(i+1), ": ", str(label_counts[i]))


# --- Run DP:
# This call runs the fit function of DPMM and also provides the "predict" function for later:
dp = DPMModel(data, alpha, prior = hyper_prior, iterations=iters, outlier_params=labels, verbose=True)

# print('results:')
# print("---")
# print(dp._labels)
# print("---")
# print(dp._k)
# print("---")
# print(dp._mu)
# print("---")
# print(dp._weights)
# print("---")
# print(dp._label_mapping)
# print("---")

#print("data 2:", data[:,2], " data 10:", data[:,10], "data 98:", data[:,98])

#x_new = np.ones((1,2))*(-1)
#print(dp.predict(x_new))
#print("---")
print(dp.predict(data.T))

# _k = len(results[1])
# _labels = results[0] - 1

# from julia import Main as jl
# jl.dpmm = results
# _d = jl.eval("dpmm[2][1].μ").shape[0] # infer d
# #_weights = jl.eval("dpmm[4]")
# _sublabels = jl.eval("dpmm[3]")

# _mu = np.empty((_k, _d))
# _sigma = np.empty((_k, _d, _d))
# _logdetsigma = np.empty(_k)
# _invsigma = np.empty((_k, _d, _d))
# _invchol = np.empty((_k, _d, _d))

# for i in range(1, _k+1):
#     _mu[i-1] = jl.eval(f"dpmm[2][{i}].μ")
#     _sigma[i-1] = jl.eval(f"dpmm[2][{i}].Σ")
#     _logdetsigma[i-1] = jl.eval(f"dpmm[2][{i}].logdetΣ")
#     _invsigma[i-1] = jl.eval(f"dpmm[2][{i}].invΣ")
#     _invchol[i-1] = jl.eval(f"dpmm[2][{i}].invChol")
#     print(_invchol[i-1])

# _det_sigma_inv_sqrt = 1/np.sqrt(np.exp(_logdetsigma))
