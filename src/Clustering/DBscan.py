from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from Clustering import Cluster
from sklearn.cluster import DBSCAN

class DBscan(Cluster):
    def __init__(self, _dataset):
        Cluster.__init__(self)
        self.dataset = _dataset.copy(deep=True)
        self.maxK = 9

    def run(self):
        data = self.getRemovedDataset()

        categorias = data['CATEGORIA'].astype("category").cat.codes.values
        especificas = data['CATEGORIA ESPECIFICA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA', 'CATEGORIA ESPECIFICA'], axis=1, inplace=True)

        keys = list(data)
        X = data.values

        index_pos = 1
        print "IMPRIMIENDO CLASIFICACION CORRECTA"
        self.plotCluster(X, categorias, keys, "Clasificacion Original", [2, 2, index_pos])
        print "LISTO"

        params = [(0.1, 10), (0.1, 20), (0.1, 50), (0.1, 80), (0.1, 100), (0.1, 500),
                  (0.3, 10), (0.3, 20), (0.3, 50), (0.3, 80), (0.3, 100), (0.3, 500),
                  (0.5, 10), (0.5, 20), (0.5, 50), (0.5, 80), (0.5, 100), (0.5, 500),
                  (0.7, 10), (0.7, 20), (0.7, 50), (0.7, 80), (0.7, 100), (0.7, 500),
                  (0.9, 10), (0.9, 20), (0.9, 50), (0.9, 80), (0.9, 100), (0.9, 500),
                  (2, 10), (2, 20), (2, 50), (2, 80), (2, 100), (2, 500),
                  (5, 10), (5, 20), (5, 50), (5, 80), (5, 100), (5, 500),
                  (10, 10), (10, 20), (10, 50), (10, 80), (10, 100), (10, 500)]

        clusters = []
        for eps, samples in params:
            clusters.append( (eps,samples, DBSCAN(eps=eps, min_samples=samples)) )

        results = []
        for eps, samples, dbscan in clusters:
            labels = dbscan.fit_predict(X)

            results.append( (labels, (eps, samples), v_measure_score(categorias, labels)) )

        results.sort(key=lambda tup: tup[2], reverse=True)

        print "IMPRIMIENDO LAS 3 MEJORES EJECUCIONES SEGUN V-SCORE"
        for i in xrange(3):
            index_pos += 1
            print "V-SCORE "+str(i+1)+":", str(results[i][2])+",", "eps = "+str(results[i][1][0])+" samples = "+str(results[i][1][1])
            self.plotCluster(X, results[i][0], keys, "DBSCAN = " + str(i+1), [2, 2, index_pos])

        self.showPlot()
