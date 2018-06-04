import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster
from Classifying import Classify
from Pca import Pca

class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #cluster1 = Cluster(self.dataset)
        #cluster1.startClusteringKMeans()

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        classify = Classify(self.dataset)
        classify.classifyBernoulliNB()

        # cluster2 = Cluster(self.dataset)
        # cluster2.startClusteringKModesFullDataHuang()
        # cluster2.startClusteringKModesFullDataCao()

        #PCA
        # pca = Pca( self.dataset )
        # pca.pca_process()

        #cluster.startClusteringKPrototypesFullDataHuang()
        # cluster.startClusteringKPrototypesMinData()

file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)
