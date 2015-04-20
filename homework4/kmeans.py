import numpy as np
import numpy.linalg
import scipy.sparse

class kmeans:

  def __init__(self,n_clusters=4, max_iter=100, tol=1e-6, verbose=False, random_seed = 1000):
   """
   This class implements k-Means clustering with incomplete data vectors.
   Arguments to the constructor include:

   n_clusters: the number of clusters to use
   max_iter: the maximum number of training iterations
   tol: convergence tolerance
   verbose: whether to provide iteration level output during fitting

   Author: Benjamin M. Marlin (marlin @cs.umass.edu)
   Date: April 7, 2015

   """

   self.K=n_clusters
   self.max_iter=max_iter
   self.tol=tol
   self.cluster_centers=[]
   self.verbose=verbose
   self.D=0
   self.random_seed = random_seed


  def item_means(self,X):
    """
    This function computes a smoothed estimate of the global mean on each dimension.
    Each dimension is smoothed against the global mean across all dimensions.
    The input is a SciPy sparse CRS matrix. The output is a numpy array.
    """
    g     = np.mean(X.data) #Compute global mean
    num   = X.sum(axis=0) + float(g) #Compute sum of values for each dimension and add global mean
    denom = (X>0).sum(axis=0) + 1.0 #Compute number of observations per dimensions
    return np.array(num/denom) #Return smoothed global mean on each dimensions

  def fit(self,X):
    """
    This function computes the cluster centers using a modified k-means implementation
    that properly accounts for incomplete feature vectors. The input is a SciPy sparse
    CRS matrix. The function sets the cluster_centers attribute of the instance.
    """

    K   = self.K      #number of clusters
    N   = X.shape[0]  #number of data cases
    D   = X.shape[1]  #number of data dimensions
    self.D=D
    np.random.seed(self.random_seed)
    mu  = 1+(np.max(X.data)-1)*np.random.rand(K,D) #Randomly initialize cluster centers
    gmu = self.item_means(X) #Compute global item means
    z   = np.zeros((N,)) #Initialize mixture indicators
    old_distortion = np.inf #Initialize distortion

    if(self.verbose):
      print("Starting KMeans learning...\n")

    #Perform optimization until max_iter iterations or convergence criterion
    #is satisfied.
    for i in range(self.max_iter):

      #Initialize accumulators for center mean numerator and denominator
      new_mu_num   = gmu*np.ones((K,D),dtype=float)
      new_mu_denom = np.ones((K,D),dtype=float)

      #Initialize current distortion
      distortion   = 0

      #Loop over data cases
      for n in range(N):
        #Compute distance to each center based on observed data
        dists = np.sum((mu[:, X[n,:].indices]-X[n,:].data)**2,axis=1)
        #Select closes cluster
        z[n] = np.argmin(dists)
        #increment distortion
        distortion += dists[z[n]]
        #Accumulate changes to cluster centers
        new_mu_num[z[n],X[n,:].indices] += X[n,:].data
        new_mu_denom[z[n],X[n,:].indices] +=1

      #Compute distortion as overall RMSE
      distortion = np.sqrt(float(distortion)/float(X.nnz))
      #Update centers based on accumulated numerators and denominators
      mu = new_mu_num/new_mu_denom
      self.cluster_centers=mu

      #Verbose iteration-level output if needed
      if(self.verbose):
        print("    Iteration %d:  Average distortion: %.7f\n"%(i,distortion))

      #Check for relative convergence
      if(np.abs(distortion-old_distortion)/distortion<self.tol):
        if(self.verbose):
          print("KMeans converged.\n")
        break;
      else:
        old_distortion = distortion

  def predict(self,X,Y):
    """
    This function computes predictions for the test items defined in Y based on the
    observations defined in X. Both are SciPy Sparse CRS matrices. The output is a
    SciPy Sparse CRS matrix of size and sparsity pattern matching Y, but containing the
    predicted rating values.
    """

    K = self.K #Number of clusters
    D = self.D #Number of data dimensions
    N = X.shape[0] #Number of cases
    mu=self.cluster_centers #Get learned centers
    #Create output matrix as copy of Y
    Yhat = scipy.sparse.csr_matrix(Y,copy=True,dtype=float)

    #Loop over all data cases
    for n in range(N):
      #Find closest center
      dists = np.linalg.norm(mu[:, X[n,:].indices]-X[n,:].data,axis=1)
      z     = np.argmin(dists)
      #Predict test item ratings using cluster mean ratings
      Yhat[n,Y[n,:].indices] = mu[z,Y[n,:].indices]
    return(Yhat)

  def cluster(self,X):
    """
    This function assigns each data case in X to the nearest cluster and
    returns an array of cluster IDs. X is a SciPy Sparse CRS matrix.
    """

    K = self.K #Number of clusters
    D = self.D #Number of data dimensions
    N = X.shape[0] #Number of cases
    mu=self.cluster_centers #Get learned centers
    z = np.zeros((N,)) #Cluster IDs

    #Loop over data
    for n in range(N):
      #Get closest cluster
      dists = np.linalg.norm(mu[:, X[n,:].indices]-X[n,:].data,axis=1)
      z[n]  = np.argmin(dists)
    return(z)

  def rmse(self,Y,Yhat):
    """
    This function computes the RMSE between the true and predicted rating
    values contained in the matrices Y and Yhat. Both must be
    SciPy Sparse CRS matrices and have identical sparsity patterns.
    """
    num   = np.sum((Y.data-Yhat.data)**2) #Compute sum of squares
    denom = Y.nnz #Get total observations
    return(np.sqrt(float(num)/float(denom))) #Return RMSE


  def get_centers(self):
    """
    This function returns the learned cluster centers as a numpy array.
    """
    return(self.cluster_centers)
