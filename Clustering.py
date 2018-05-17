import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, euclidean

class Cluster:
    def __init__(self, dataset):
        self.dataset = dataset
        self.maxK = 10

    def startClustering(self):
        categorias = self.dataset['CATEGORIA'].astype("category").cat.codes.values
        self.dataset.drop(['CATEGORIA'], axis=1, inplace=True)

        self.fignum = 1

        keys = list(self.dataset)
        X = self.dataset.values

        clusters = []

        for i in xrange( 3, self.maxK + 1 ):
            clusters.append( (i, KMeans(n_clusters=i)) )

        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster( X, categorias, [0,0,0,0], keys, "Clasificacion original" )
        print "LISTO"

        print "CALCULANDO EJECUCIONES K MEANS"
        errors = []
        dbindexes = []
        for i, kmeans in clusters:
            print "CALCULANDO K =", i
            kmeans.fit(X)
            labels = kmeans.predict(X)
            centers = kmeans.cluster_centers_

            print "\tError cuadratico:", kmeans.inertia_
            errors.append( kmeans.inertia_ )
            dbIndex = self.daviesbouldin(X, labels, centers)
            print "\tIndice Davies-Bouldin:", dbIndex

            dbindexes.append(dbIndex)

            self.plotCluster( X, labels, centers, keys, "# Clusters: " + str(len(centers)) )

        print "LISTO"

        print "GRAFICANDO ERROR CUADRATICO"
        k_X = np.array( xrange(3, self.maxK+1 ) )
        sse_Y = np.array( errors )
        db_Y = np.array(dbindexes)


        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        plt.show()
        plt.plot(k_X, db_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Indice Davies-Bouldin")
        plt.show()
        print "LISTO"

    def plotCluster(self, data, labels, centers, keys, title):
        fig = plt.figure(self.fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel(keys[2])
        ax.set_title(title)
        ax.dist = 12

        self.fignum += 1
        fig.show()

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
        return np.mean(DB_indexes)