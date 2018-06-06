from __future__ import division, print_function
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from src.Preprocessing.Preprocessing import Preprocess
from src.Preprocessing.Discretizing import Discretize
from src.Association.Association import Association
from src.Clustering.Kmodes import Kmodes
from src.Classifying.Runner import Runner

import weka.core.jvm as jvm


class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #kmeans = Kmeans(self.dataset)
        #kmeans.run()

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        # Association
        association = Association(self.dataset)
        keys = ["HORAS AL MES", "DIVISION", "AREA DE PERSONAL", "SEXO", "EDAD DEL EMPLEADO", "SALARIOS MINIMOS",
                "CATEGORIA"]
        association.apriori(keys, confidence=0.7)
        association.filteredApriori(keys, confidence=0.7)
        jvm.stop()

        kmodes = Kmodes(self.dataset)
        # kmodes.startClusteringKModesFullDataHuang()
        kmodes.startClusteringKModesFullDataCao()

        # cluster.startClusteringKMeans()

        # cluster.startClusteringKModesFullDataHuang()
        # cluster.startClusteringKModesFullDataCao()

        # cluster.startClusteringKPrototypesFullData()
        # cluster.startClusteringKPrototypesMinData()

        # cluster1 = Cluster(self.dataset)
        # cluster1.startClusteringKMeans()

        # cluster2 = Cluster(self.dataset)
        # cluster2.startClusteringKModesFullDataHuang()
        # cluster2.startClusteringKModesFullDataCao()

        # Classifying
        classifier = Runner(self.dataset)
        classifier.runClassifiers()

        #PCA
        # pca = Pca( self.dataset )
        # pca.pca_process()




file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)
