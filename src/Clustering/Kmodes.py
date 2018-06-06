import numpy as np
import matplotlib.pyplot as plt
from Clustering import Cluster
from kmodes.kmodes import KModes

from Clustering import Cluster

class Kmodes(Cluster):
    def __init__(self, _dataset):
        Cluster.__init__(self)
        self.dataset = _dataset.copy(deep=True)
        self.maxK = 9

    def getRemovedDataset(self):
        data = self.dataset.copy(deep=True)
        erase_vars = [ 'FECHA INICIO POSESION', 'TURNO', 'HORAS AL MES', 'HORARIO TRABAJO',
                      'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO',
                      'AREA DE NOMINA', 'CENTRO DE COSTE',
                      'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO',
                      'ROL DEL EMPLEADO', 'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR']

        data = self.dataset.drop(erase_vars, axis=1)
        data = data[[ 'EDAD DEL EMPLEADO', 'SALARIOS MINIMOS', 'AFILIADO A PAC', 'CATEGORIA ESPECIFICA', 'CATEGORIA']]

        return data

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
        self.plotCluster(C, categorias, keys, "Clasificacion Original", [2, 4, index_pos])
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
            errors.append(k_modes.cost_)

            index_pos += 1
            self.plotCluster(C, labels, keys, "K = " + str(len(centers)), [2,4,index_pos])
        print "LISTO"

        plt.show()
        plt.clf()

        print "GRAFICANDO INDICES"
        k_X = np.array(xrange(3, self.maxK + 1))
        sse_Y = np.array(errors)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        plt.show()
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
            centers = k_modes.cluster_centroids_

            print "\tError cuadratico:", k_modes.cost_
            print "\t# iteraciones:", k_modes.n_iter_
            errors.append(k_modes.cost_)
            index_pos += 1
            self.plotCluster(C, labels, keys, "K = " + str(len(centers)), [2,4,index_pos])
        print "LISTO"

        plt.show()
        plt.clf()

        print "GRAFICANDO INDICES"
        k_X = np.array(xrange(3, self.maxK + 1))
        sse_Y = np.array(errors)

        plt.plot(k_X, sse_Y, '-o')
        plt.xlabel("# de clusters (k)")
        plt.ylabel("Suma de distancias cuadradas")
        plt.show()
        print "LISTO"