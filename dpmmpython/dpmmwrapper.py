import numpy as np
from scipy.stats import multivariate_normal as gaussian

import julia
julia.install()

from dpmmpython.priors import niw, multinomial
from julia import DPMMSubClusters


class DPMMPython:
    """
     Wrapper for the DPMMSubCluster Julia package
     """


    @staticmethod
    def create_prior(dim,mean_prior,mean_str,cov_prior,cov_str):
        """
        Creates a gaussian prior, if cov_prior is a scalar, then creates an isotropic prior scaled to that, if its a matrix
        uses it as covariance
        :param dim: data dimension
        :param mean_prior: if a scalar, will create a vector scaled to that, if its a vector then use it as the prior mean
        :param mean_str: prior mean psuedo count
        :param cov_prior: if a scalar, will create an isotropic covariance scaled to cov_prior, if a matrix will use it as
        the covariance.
        :param cov_str: prior covariance psuedo counts
        :return: DPMMSubClusters.niw_hyperparams prior
        """
        if isinstance(mean_prior,(int,float)):
            prior_mean = np.ones(dim) * mean_prior
        else:
            prior_mean = mean_prior

        if isinstance(cov_prior, (int, float)):
            prior_covariance = np.eye(dim) * cov_prior
        else:
            prior_covariance = cov_prior
        prior =niw(mean_str,prior_mean,dim + cov_str, prior_covariance)
        return prior


    @staticmethod
    def fit(data,alpha, prior = None,
            iterations= 100, verbose = False,
            burnout = 15, gt = None, outlier_weight = 0, outlier_params = None):
        """
        Wrapper for DPMMSubClusters fit, refer to "https://bgu-cs-vil.github.io/DPMMSubClusters.jl/stable/usage/" for specification
        Note that directly working with the returned clusters can be problematic software displaying the workspace (such as PyCharm debugger).
        :return: labels, clusters, sublabels, renormalized weights
        """
        if prior == None:
            results = DPMMSubClusters.fit(data,alpha, iters = iterations,
                                          verbose = verbose, burnout = burnout,
                                          gt = gt, outlier_weight = outlier_weight,
                                          outlier_params = outlier_params)
        else:
            results = DPMMSubClusters.fit(data, prior.to_julia_prior(), alpha, iters=iterations,
                                          verbose=verbose, burnout=burnout,
                                          gt=gt, outlier_weight=outlier_weight,
                                          outlier_params=outlier_params)
            
        weights = results[2] / np.sum(results[2])
        return results[0],results[1],results[-1],weights


    @staticmethod
    def get_model_ll(points,labels,clusters):
        """
        Wrapper for DPMMSubClusters cluster statistics
        :param points: data
        :param labels: labels
        :param clusters: vector of clusters distributions
        :return: vector with each cluster avg ll
        """
        return DPMMSubClusters.cluster_statistics(points,labels,clusters)[0]


    @staticmethod
    def add_procs(procs_count):
        j = julia.Julia()
        j.eval('using Distributed')
        j.eval('addprocs(' + str(procs_count) + ')')
        j.eval('@everywhere using DPMMSubClusters')
        j.eval('@everywhere using LinearAlgebra')
        j.eval('@everywhere BLAS.set_num_threads(2)')

    
    @staticmethod
    def generate_gaussian_data(sample_count,dim,components,var):
        '''
        Wrapper for DPMMSubClusters cluster statistics
        :param sample_count: how much of samples
        :param dim: samples dimension
        :param components: number of components
        :param var: variance between componenets means
        :return: (data, gt)
        '''
        data = DPMMSubClusters.generate_gaussian_data(sample_count, dim, components, var)
        gt =  data[1]
        data = data[0]
        return data,gt


class DPMModel(DPMMPython):
    
    """
    Wrapper class for DPMMSubClusters results. Naive implementation of 
    predicted labels and responsibilities for new observations.
    
    (It's naive because it uses scipy's multivariate normal class 
    rather than making use of information about the determinant and
    Cholesky decomposition of the precision matrix that are stored in
    the mv_gaussian objects that come back with DPMMSubClusters.)
    """


    def predict(self, X: np.array) -> np.array:
        """
        Return most likely label for each observation in new data X
        
        X :: NxD where N is number of new observations and D is dimensionality
        
        Returns an Nx1 array of most likely labels
        n.b.: Results generated in Julia have labels that begin at 1, 
        labels must be adjusted before export or this won't be correct
        """
        
        
        if len(X.shape) == 1:
            if X.shape[0] == self._d:
                # to conform to expected output from sklearn
                return np.array([np.argmax(self.predict_proba(X))])
            else:
                raise Exception
        elif X.shape[1] != self._d:
            raise Exception
        else:
            return np.argmax(self.predict_proba(X), 1)
  
        
    def predict_proba(self, X: np.array) -> np.array:
        """ 
        Return responsibilites of components conditional on new data X
        
        X :: NxD where N is number of new observations and D is dimensionality
        
        Returns an NxK array where K is the number of components
        
        """
        
        resp = np.array([((np.log(self._weights[k]) + 
                           self._dists[k].logpdf(X)) -
                           np.log(np.sum([self._weights[i] * 
                                          self._dists[i].pdf(X) for i
                                          in range(self._k)], 0)
                                          ))
                           for k in range(self._k)]).T
        
        return np.exp(resp)
    
        
    def __init__(self, *args, **kwargs):
        """
        dpmm: output from DPMMPython.fit()
        """
        
        from julia import Main as jl # Objects attached -> use local namespace
        if 'seed' in kwargs:
            from julia import Random
            self.seed = kwargs.pop('seed')
            jl.eval(f"Random.seed!({self.seed})"); 
        
        fitted = self.fit(*args, **kwargs)
        # adjust labels for python, where indexes begin at 0
        self._labels = fitted[0] - 1
        self._k = len(fitted[1]) # infer k
        
        jl.dpmm = fitted
        
        self._d = jl.eval("dpmm[2][1].μ").shape[0] # infer d
        self._dists = [gaussian(jl.eval(f"dpmm[2][{i}].μ"), 
                                jl.eval(f"dpmm[2][{i}].Σ")) for i in range(1, self._k+1)]
        self._weights = jl.eval("dpmm[4]")
        self._sublabels = jl.eval("dpmm[3]")
        
        
if __name__ == "__main__":
    j = julia.Julia()
    data,gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
    labels,_,sub_labels= DPMMPython.fit(data,100,prior = prior,verbose = True, gt = gt)
    prior = 0
    _ = 0
    print(labels)
