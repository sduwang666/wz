print(__doc__)

from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, cluster
#引入算法
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#引入数据集
from sklearn.datasets import load_digits

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

#算法
# km=KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
# ap=AffinityPropagation(),
# ms=MeanShift(),
# bd=estimate_bandwidth(),
# sc=SpectralClustering(),
# ac=AgglomerativeClustering(),
# wd=AgglomerativeClustering(),
# dc=DBSCAN()



print(82 * '_')
print('init\t\ttime\tnorm\thomo\tcomp')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)

    print('%-5s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based", data=data)
bench_k_means(AffinityPropagation(),name="AP", data=data)
bench_k_means(MeanShift(),name="MS", data=data)
bench_k_means(AgglomerativeClustering(n_clusters=n_digits),name="AC", data=data)
bench_k_means(AgglomerativeClustering(n_clusters=n_digits),name="WD", data=data)
bench_k_means(DBSCAN(),name="DB", data=data)
bench_k_means(SpectralClustering(n_clusters=n_digits),name="SC", data=data)

GM=GaussianMixture(n_components=10)
t0 = time()
GM.fit(data)
print('%-5s\t%.2fs\t%.3f\t%.3f\t%.3f'
     % ('GM', (time() - t0),metrics.normalized_mutual_info_score(labels, GM.predict(data)),
             metrics.homogeneity_score(labels, GM.predict(data)),
             metrics.completeness_score(labels, GM.predict(data))))

print(82 * '_')





































# #############################################################################
# Visualize the results on PCA-reduced data
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()