'''
Created on Sep 6, 2017

@author: mroch
'''

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
        self.N = data.shape[0]
        self.dimensions = data.shape[1]
        self.varcovar = np.cov(self.data)
        pca = PCA(data)


        # You are not required to implement corr_anal == True case
        # but it's pretty easy to do once you have the variance-
        # covariance case done
        
    def get_pca_directions(self):
        """get_pca_directions() - Return matrix of PCA directions
        Each column is a PCA direction, with the first column
        contributing most to the overall variance.
        """

        return pca.get_pca_directions()

             
    def transform(self, data, dim=None):
        """transform(data, dim) - Transform data into PCA space
        To reduce the dimension of the data, specify dim as the remaining 
        number of dimensions. Omitting dim results in using all PCA axes 
        """
        
        
    def get_component_loadings(self):
        """get_component_loadings()
        Return a square matrix of component loadings. Column j shows the amount
        of variance from each variable i in the original space that is accounted
        for by the jth principal component
        """

        
        
if __name__ == '__main__':
    
    plt.ion()  # Interactive plotting

    # Demonstration of using PCA

    # Generate synthetic data set and plot it
    (data, mu, Sigma) = genData(500, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    h_data = ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('$f_0$')
    ax.set_ylabel('$f_1$')
    ax.set_zlabel('$f_2$')

    # Conduct PCA analysis using variance-covariance method
    pca = PCA(data)

    # Show the PCA directions
    k = 10  # scale up unit vectors by k
    v = pca.get_pca_directions()
    for idx in range(v.shape[1]):
        ax.quiver(0, 0, 0, k * v[idx, 0], k * v[idx, 1], k * v[idx, 2],
                  color='xkcd:orange')

    # project onto 2d
    vc_proj = pca.transform(data, 2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    h_vc_proj = ax.scatter(vc_proj[:, 0], vc_proj[:, 1], color='xkcd:orange')


    # repeat w/ autocorrelation analysis
    pcaR = PCA(data, corr_anal=True)

    w = pcaR.get_pca_directions()
    for idx in range(v.shape[1]):
        ax.quiver(0, 0, 0, k * w[idx, 0], k * w[idx, 1], k * w[idx, 2], color='xkcd:lilac')

    # project onto 2d
    r_proj = pcaR.transform(data, 2)
    h_r_proj = ax.scatter(r_proj[:, 0], r_proj[:, 1], color='xkcd:lilac')

    # Add legend.  $...$ enables LaTeX-like equation formatting
    plt.legend([h_data, h_vc_proj, h_r_proj],
               ['Original data', '$\Sigma $projection', '$R$ projection'])
    print("Autocorrelation component loadings")
    print(pcaR.get_component_loadings())

    x = 3  # breakable line so our windows don't go away
    
