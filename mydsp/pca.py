'''
Created on Sep 6, 2017

@author: mroch
'''
import scipy
import numpy as np
import scipy.signal as sig

def genData(N, dim):
    """genData(N, dim) - Return N samples of dim dimensional normal data at random positions
        returns(data, mu, var)  where mu and var are the mu/var for each dimension
    """
    # Generate random parameters 
    mu = np.random.rand(dim) * 50
    mu[2] = mu[2] + 70  # Raise z mean so above projection
    var = np.array([50, 100, 5]) + np.random.rand(dim) * 10
        
    
    # Build up random variance-covariance matrix
    varcov = np.zeros([dim, dim])
    # Fill off diagonals in lower triangle
    for i in range(1, dim):
        varcov[i, 0:i] = np.random.rand(i) * 5
    # make symmetric
    varcov = varcov + np.transpose(varcov)
    # add in variances
    varcov = varcov + np.diag(var)
    
    data = np.random.multivariate_normal(mu, varcov, N)
    return (data, mu, varcov)
        
    
    

class PCA(object):
    '''
    PCA
    '''


    def __init__(self, data, corr_anal=False):
        '''
        PCA(data)
        data matrix is N x Dim
        Performs variance-covariance matrix or correlation matrix
        based principal components analysis of detrended (by
        mean removal) data.
        
        If the optional corr_anal is True, correlation-based PCA
        is performed
        '''

        self.data = data
        self.N = np.asarray(data).shape[0]
        self.dimensions = np.asarray(data).shape[1]
        self.data_std = sig.detrend(self.data,axis=0)
        self.varcovar = np.cov(self.data_std, rowvar = False)

        self.eig_vals, self.eig_vecs = scipy.linalg.eig(self.varcovar)

        eig_vals_index = np.flip(np.argsort(self.eig_vals))

        eig_vec_sorted = []
        for i in eig_vals_index:
            eig_vec_sorted.append(self.eig_vals[eig_vals_index])

        self.eig_vecs = eig_vec_sorted
        # You are not required to implement corr_anal == True case
        # but it's pretty easy to do once you have the variance-
        # covariance case done
        
    def get_pca_directions(self):
        """get_pca_directions() - Return matrix of PCA directions
        Each column is a PCA direction, with the first column
        contributing most to the overall variance.
        """
        return self.eig_vecs

    def get_eig_vals(self):
        '''get_eig_vals(): Return a matrix with eigen values'''

        return self.eig_vals
             
    def transform(self, data, dim=None):
        """transform(data, dim) - Transform data into PCA space
        To reduce the dimension of the data, specify dim as the remaining 
        number of dimensions. Omitting dim results in using all PCA axes 
        """
        self.eig_vec = np.transpose(self.get_pca_directions())
        if dim is None:
            return np.dot(self.data, np.asarray(self.get_pca_directions()))
        else:
            return np.dot(self.data, np.asarray(self.get_pca_directions())[:, 0:dim])

    def get_component_loadings(self):
        """get_component_loadings()
        Return a square matrix of component loadings. Column j shows the amount
        of variance from each variable i in the original space that is accounted
        for by the jth principal component
        """
        return np.divide(
            np.dot(np.asarray(self.eig_vecs), np.asarray(np.sqrt(self.eig_vals))),
            self.data_std)

        


    
