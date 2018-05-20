import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, euclidean
from kmodes.kprototypes import KPrototypes

class Cluster:
    def __init__(self, dataset):
        self.dataset = dataset
        self.maxK = 15

    def plotCluster(self, data, labels, centers, keys, title, folder):
        fig = plt.figure(self.fignum, figsize=(8, 7))
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

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
        fig.savefig( folder+title+".png" )
        fig.clf()

    def startClusteringKMeans(self):
        erase_vars = ['CARGO', 'FECHA INICIO POSESION', 'FIN', 'TURNO',
                      'SUELDO TEXTO', 'SALARIO', 'HORAS AL MES', 'HORAS SEMANALES', 'HORAS DIARIAS',
                      'HORARIO TRABAJO', 'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL',
                      'TIPO DE PACTO', 'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA',
                      'CENTRO DE COSTE', 'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL',
                      'AREA DE PERSONAL', 'SEXO', 'ROL DEL EMPLEADO', 'SALARIO A 240',
                      'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR' ]

        data = self.dataset.drop(erase_vars, axis=1)

        data = data[['SALARIOS MINIMOS', 'EDAD DEL EMPLEADO', 'AFILIADO A PAC', 'CATEGORIA ESPECIFICA', 'CATEGORIA']]

        # data['AFILIADO A PAC'] = map(lambda x: 0 if x == 1 else 1, data['AFILIADO A PAC'])

        self.fignum = 1

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        especificas = data['CATEGORIA ESPECIFICA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)

        keys = list(data)
        X = data.values

        folder = "results/clustering/kmeans/"

        clusters = []

        for i in xrange( 3, self.maxK + 1 ):
            clusters.append( (i, KMeans(n_clusters=i)) )

        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster(X, categorias, [0, 0, 0, 0], keys, "Clasificacion Original", folder)
        self.plotCluster(X, especificas, [0, 0, 0, 0, 0, 0, 0, 0], keys, "Clasificacion especifica Original", folder )
        print "LISTO"

        print "CALCULANDO EJECUCIONES K MEANS"
        errors = []
        dbindexes = []
        file = open( folder + "/results.txt", "w+" )
        file.write("Resultados ejecucion K-means\n\n")

        file.write("Variables:\n")
        file.write(str(keys) + "\n\n")
        
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

            file.write("K = " + str(i) + "\n")
            file.write("\tSuma de distancias cuadradas: " + str(kmeans.inertia_) + "\n")
            file.write("\tIndice Davies-Bouldin: " + str(dbIndex) + "\n\n")

            self.plotCluster( X, labels, centers, keys, "K = " + str(len(centers)), folder + "Ks/" )

        print "LISTO"

        file.close()

        print "GRAFICANDO INDICES"
        k_X = np.array( xrange(3, self.maxK+1 ) )
        sse_Y = np.array( errors )
        db_Y = np.array(dbindexes)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        plt.savefig( folder + "Suma distancias.png")
        plt.clf()

        plt.plot(k_X, db_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Indice Davies-Bouldin")
        plt.savefig( folder + "Davies Bouldin.png" )
        plt.clf()
        print "LISTO"

    def startClusteringKPrototypesFullData(self):
        data = self.dataset

        erase_vars = ['FECHA INICIO POSESION', 'FIN', 'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO', 'CATEGORIA ESPECIFICA']

        data = data.drop(erase_vars, axis=1)

        data['AFILIADO A PAC'] = data['AFILIADO A PAC'].astype("category")
        data['FAMILIAR AFILIADO A PAC'] = data['FAMILIAR AFILIADO A PAC'].astype("category")
        data['ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR'] = data['ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR'].astype("category")

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA'], axis=1, inplace=True)

        keys = list(data)
        X = data.values

        print "CORRIENDO K-PROTOTYPES CON K = 4"
        k_prot = KPrototypes( n_clusters=4, init='Cao', verbose=0 )
        clusters = k_prot.fit_predict(X, categorical=[0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25])
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
        return np.mean(DB_indexes)