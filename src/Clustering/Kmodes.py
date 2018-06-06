import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from Clustering import Cluster

class Kmodes(Cluster):
    def __init__(self, _dataset):
        Cluster.__init__(self)
        self.dataset = _dataset.copy(deep=True)
        self.maxK = 9

    def startClusteringKModesFullDataHuang(self):
        data = self.dataset.copy(deep=True)
        cleaned = self.getRemovedDataset()

        cleaned['SALARIOS MINIMOS'] = cleaned['SALARIOS MINIMOS'].astype("category").cat.codes.values
        cleaned['EDAD DEL EMPLEADO'] = cleaned['EDAD DEL EMPLEADO'].astype("category").cat.codes.values
        cleaned['AFILIADO A PAC'] = cleaned['AFILIADO A PAC'].astype("category").cat.codes.values

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)
        cleaned.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)

        keys = list(cleaned)
        X = data.values
        C = cleaned.values
        clusters = []
        index_pos = 1

        plt.clf()
        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster(C, categorias, keys, "Clasificacion Original", [1, 2, index_pos])
        print "LISTO"

        for i in xrange(3, self.maxK + 1):
            clusters.append((i, KModes(n_clusters=i, init='Huang', verbose=0)))

        print "CALCULANDO EJECUCIONES K-MODES HUANG"
        errors = []

        for i, k_modes in clusters:
            print "CALCULANDO K =", i
            labels = k_modes.fit_predict(X)
            centers = k_modes.cluster_centroids_

            print "\tError cuadratico:", k_modes.cost_
            print "\t# iteraciones:", k_modes.n_iter_

            scores = self.getScores( categorias, labels )
            print "\tHomogeneidad:", scores[0]
            print "\tCompletitud:", scores[1]
            print "\tV-score:", scores[2]

            errors.append(k_modes.cost_)

            index_pos += 1
            self.plotCluster(C, labels, keys, "K = " + str(len(centers)), [2,4,index_pos])
        print "LISTO"

        self.showPlot()

        print "GRAFICANDO INDICES"
        k_X = np.array(xrange(3, self.maxK + 1))
        sse_Y = np.array(errors)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        self.showPlot()
        print "LISTO"

    def startClusteringKModesFullDataCao(self):
        data = self.dataset.copy(deep=True)
        cleaned = self.getRemovedDataset()

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

        index_pos = 1
        plt.clf()
        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster(C, categorias, keys, "Clasificacion Original", [2,4,index_pos])
        print "LISTO"

        for i in xrange(3, self.maxK + 1):
            clusters.append((i, KModes(n_clusters=i, init='Cao', verbose=0)))

        print "CALCULANDO EJECUCIONES K-MODES CAO"
        errors = []

        for i, k_modes in clusters:
            print "CALCULANDO K =", i
            labels = k_modes.fit_predict(X)

            print "\tError cuadratico:", k_modes.cost_
            print "\t# iteraciones:", k_modes.n_iter_

            scores = self.getScores( categorias, labels )
            print "\tHomogeneidad:", scores[0]
            print "\tCompletitud:", scores[1]
            print "\tV-score:", scores[2]

            errors.append(k_modes.cost_)
            index_pos += 1
            self.plotCluster(C, labels, keys, "K = " + str(i), [2,4,index_pos])
        print "LISTO"

        self.showPlot()

        print "GRAFICANDO INDICES"
        k_X = np.array(xrange(3, self.maxK + 1))
        sse_Y = np.array(errors)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        self.showPlot()
        print "LISTO"