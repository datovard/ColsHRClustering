import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, euclidean
from Clustering import Cluster
from sklearn.cluster import KMeans

class Kmeans(Cluster):
    def __init__(self, _dataset):
        Cluster.__init__(self)
        self.dataset = _dataset.copy(deep=True)
        self.maxK = 9

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

    def run(self):
        data = self.getRemovedDataset()

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        especificas = data['CATEGORIA ESPECIFICA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)

        keys = list(data)
        X = data.values

        clusters = []
        index_pos = 1
        for i in xrange(3, self.maxK + 1):
            clusters.append((i, KMeans(n_clusters=i)))


        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster(X, categorias, keys, "Clasificacion Original", [2,4,index_pos])
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
            errors.append(kmeans.inertia_)
            dbIndex = self.daviesbouldin(X, labels, centers)
            scores = self.getScores( categorias, labels )
            print "\tIndice Davies-Bouldin:", dbIndex
            print "\tHomogeneidad:", scores[0]
            print "\tCompletitud:", scores[1]
            print "\tV-score:", scores[2]

            dbindexes.append(dbIndex)

            index_pos += 1
            self.plotCluster(X, labels, keys, "K = " + str(len(centers)), [2,4,index_pos] )
        print "LISTO"

        self.showPlot()

        print "GRAFICANDO INDICES"
        k_X = np.array(xrange(3, self.maxK + 1))
        sse_Y = np.array(errors)
        db_Y = np.array(dbindexes)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        self.showPlot()

        plt.plot(k_X, db_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Indice Davies-Bouldin")
        self.showPlot()
        print "LISTO"