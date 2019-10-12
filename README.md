# **wz**

## Comparison of various algorithms

KMeans:The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. It scales well to large number of samples and has been used across a large range of application areas in many different fields.

AffinityPropagation:AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one
sample to be the exemplar of the other, which is updated in response to the
values from other pairs. This updating happens iteratively until convergence,
at which point the final exemplars are chosen, and hence the final clustering
is given.

MeanShift:MeanShift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

SpectralClustering:SpectralClustering does a low-dimension embedding of the
affinity matrix between samples, followed by a KMeans in the low dimensional space. It is especially efficient if the affinity matrix is sparse and the [pyamg](https://github.com/pyamg/pyamg) module is installed.SpectralClustering requires the number of clusters to be specified. It works well for a small number of clusters but is not advised when using many clusters.

Hierarchical clustering:Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the
clusters with only one sample. 

AgglomerativeClustering:The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together.

DBSCAN:The DBSCAN algorithm views clusters as areas of high density
separated by areas of low density. Due to this rather generic view, clusters
found by DBSCAN can be any shape, as opposed to k-means which assumes that
clusters are convex shaped. The central component to the DBSCAN is the concept
of *core samples*, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other(measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples). There are two parameters to the algorithm,min_samples and eps,which define formally what we mean when we say *dense*.Higher min_samples or lower eps indicate higher density necessary to form a cluster.

Gaussian mixture:A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of
Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

------

## homework 1

**result:**

D:\pycharm_workspace\venv\Scripts\python.exe D:/pycharm_workspace/homework.py
None
n_digits: 10, 	 n_samples 1797, 	 n_features 64

__________________________________________________________________________________

init		            time	 norm	homo	comp  
k-means++	  0.25s	0.626	0.602	0.650  
//random	        0.14s	0.689	0.669	0.710  
//PCA-based	  0.02s	0.681	0.667	0.695  
AP   	              8.06s	0.655	0.932	0.460  
MS   	   		  5.78s	0.063	0.014	0.281  
AC   				  0.20s	0.797	0.758	0.836  
//WD   				0.16s	0.797	0.758	0.836  
DB   				  0.31s	0.375	0.000	1.000  
SC   				  599.36s  0.012	0.001	0.271  
GM   				 0.69s	0.630	0.596	0.665  

__________________________________________________________________________________

Process finished with exit code 0

**conclusion:**

​            1.The shortest time is the "AgglomerativeClustering" algorithm.

2. "AgglomerativeClustering" algorithm is optimal under NMI evaluation index
3. "AffinityPropagation" algorithm is optimal under  Homogeneity evaluation index
4. "DBSCAN" algorithm is optimal under  Completeness evaluation index

__________________________________________________________________________________

## homework 2

**result:**

D:\pycharm_workspace\venv\Scripts\python.exe D:/pycharm_workspace/homework2.py
None
Usage: homework2.py [options]

Options:
  -h, --help            show this help message and exit
  --lsa=N_COMPONENTS    Preprocess documents with latent semantic analysis.
  --no-minibatch        Use ordinary k-means algorithm (in batch mode).
  --no-idf              Disable Inverse Document Frequency feature weighting.
  --use-hashing         Use a hashing feature vectorizer
  --n-features=N_FEATURES
                        Maximum number of features (dimensions) to extract
                        from text.
  --verbose             Print progress reports inside k-means algorithm.
Loading 20 newsgroups dataset for categories:
['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
3387 documents
4 categories

Extracting features from the training dataset using a sparse vectorizer
done in 0.851977s
n_samples: 3387, n_features: 10000

Clustering sparse data with MiniBatchKMeans(batch_size=1000, compute_labels=True, init='k-means++',
                init_size=1000, max_iter=100, max_no_improvement=10,
                n_clusters=4, n_init=1, random_state=None,
                reassignment_ratio=0.01, tol=0.0, verbose=False)  
time: 0.084s  
Homogeneity: 0.490  
Completeness: 0.490  
normalized_mutual_info: 0.490

Clustering sparse data with AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
                    damping=0.5, max_iter=200, preference=None, verbose=False)  
time: 19.101s  
Homogeneity: 0.885  
Completeness: 0.191  
normalized_mutual_info: 0.411

Clustering sparse data with MeanShift(bandwidth=1.3966928550906759, bin_seeding=True, cluster_all=True,
          min_bin_freq=1, n_jobs=None, seeds=None)  
time: 32.626s  
Homogeneity: 0.000  
Completeness: 1.000  
normalized_mutual_info: 0.000

Clustering sparse data with SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=8, n_init=10, n_jobs=None,
                   n_neighbors=10, random_state=None)
time: 2.697s  
Homogeneity: 0.398  
Completeness: 0.379  
normalized_mutual_info: 0.388

Clustering sparse data with AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='ward', memory=None, n_clusters=2,
                        pooling_func='deprecated')  
time: 56.186s  
Homogeneity: 0.342  
Completeness: 0.698  
normalized_mutual_info: 0.489

Clustering sparse data with DBSCAN(algorithm='auto', eps=0.5, leaf_size=30, metric='euclidean',
       metric_params=None, min_samples=5, n_jobs=None, p=None)  
time: 0.446s  
Homogeneity: 0.002  
Completeness: 0.168  
normalized_mutual_info: 0.016

Clustering sparse data with GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                means_init=None, n_components=1, n_init=1, precisions_init=None,
                random_state=None, reg_covar=1e-06, tol=0.001, verbose=0,
                verbose_interval=10, warm_start=False, weights_init=None)  
time: 80.286s  
Homogeneity: 0.000  
Completeness: 1.000  
normalized_mutual_info: 0.000

Top terms per cluster:
Cluster 0: space nasa henry access toronto digex alaska gov pat shuttle
Cluster 1: graphics image com file files thanks software gif 3d university
Cluster 2: god com sandvik people jesus don say sgi morality think
Cluster 3: uk com university posting nntp host article cwru cs ac

Process finished with exit code 0

**conclusion:**

​       	 1.The shortest time is the "k-means++" algorithm.

2. "k-means++" algorithm is optimal under NMI evaluation index
3. "AffinityPropagation" algorithm is optimal under  Homogeneity evaluation index
4. "GaussianMixture" algorithm is optimal under  Completeness evaluation index
