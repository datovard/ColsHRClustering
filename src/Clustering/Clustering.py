import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, euclidean
from kmodes.kprototypes import KPrototypes


class Cluster:
    def __init__(self):
        self.startPlotCluster()

    def startPlotCluster(self):
        self.fig = plt.figure(1, figsize=(8, 7))

    def plotCluster(self, data, labels, keys, title, subplot ):
        ax = self.fig.add_subplot(subplot[0], subplot[1], subplot[2], projection="3d") #Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels.astype(np.float), edgecolor='k', s=50 )

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(keys[0], labelpad=-10).set_size(7)
        ax.set_ylabel(keys[1], labelpad=-10).set_size(7)
        ax.set_zlabel(keys[2], labelpad=-10).set_size(7)
        ax.set_title(title)
        ax.dist = 12

    '''
    def startClusteringKPrototypesFullDataHuang(self):
        data = self.dataset
        cleaned = self.getRemovedDataset()

        folder = "results/clustering/kprototypes/"

        self.fignum = 1

        cleaned['SALARIOS MINIMOS'] = cleaned['SALARIOS MINIMOS'].astype("category").cat.codes.values
        cleaned['EDAD DEL EMPLEADO'] = cleaned['EDAD DEL EMPLEADO'].astype("category").cat.codes.values
        cleaned['AFILIADO A PAC'] = cleaned['AFILIADO A PAC'].astype("category").cat.codes.values

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        especificas = data['CATEGORIA ESPECIFICA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)
        cleaned.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)

        keys = list(cleaned)
        X = data.values
        C = cleaned.values
        clusters = []

        full = list(data)

        print "CORRIENDO K-PROTOTYPES CON K = 4"
        k_prot = KPrototypes( n_clusters=4, init='Cao', verbose=0 )

        clusters = k_prot.fit_predict(X, categorical=range(26))
        centroids = k_prot.cluster_centroids_

        bad = 0; good =0
        for i in xrange(len(clusters)):
            if clusters[i] == categorias[i]:
                good += 1
            else:
                bad += 1

        print good, bad
        print (float(good)/len(clusters))*100.0, (float(bad)/len(clusters))*100.0

    def daviesbouldin( self, X, labels, centroids ):
        nbre_of_clusters = len(centroids)  # Get the number of clusters
        distances = [[] for e in range(nbre_of_clusters)]  # Store intra-cluster distances by cluster
        distances_means = []  # Store the mean of these distances
        DB_indexes = []  # Store Davies_Boulin index of each pair of cluster
        second_cluster_idx = []  # Store index of the second cluster of each pair
        first_cluster_idx = 0  # Set index of first cluster of each pair to 0

        # Step 1: Compute euclidean distances between each point of a cluster to their centroid
        for cluster in range(nbre_of_clusters):
            for point in range(X[labels == cluster].shape[0]):
                distances[cluster].append(euclidean(X[labels == cluster][point], centroids[cluster]))

        # Step 2: Compute the mean of these distances
        for e in distances:
            distances_means.append(np.mean(e))

        # Step 3: Compute euclidean distances between each pair of centroid
        ctrds_distance = pdist(centroids)

        # Tricky step 4: Compute Davies-Bouldin index of each pair of cluster
        for i, e in enumerate(e for start in range(1, nbre_of_clusters) for e in range(start, nbre_of_clusters)):
            second_cluster_idx.append(e)
            if second_cluster_idx[i - 1] == nbre_of_clusters - 1:
                first_cluster_idx += 1
            DB_indexes.append((distances_means[first_cluster_idx] + distances_means[e]) / ctrds_distance[i])

        # Step 5: Compute the mean of all DB_indexes
        return np.mean(DB_indexes)'''