import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster
from Pca import Pca
from DecisionTree import DecisionTree

class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #Discretize
        discretize = Discretize( self.dataset, True )
        self.dataset = discretize.discretizeFile()

        #PCA
        # pca = Pca( self.dataset )
        # pca.pca_process()

        #DecisionTree
        #decisionTree = DecisionTree( self.dataset )
        #decisionTree.run()


        #Cluster
        # cluster = Cluster( self.dataset )
        # cluster.startClusteringKMeans()

        # cluster.startClusteringKModesFullDataHuang()
        # cluster.startClusteringKModesFullDataCao()

        # cluster.startClusteringKPrototypesFullData()
        # cluster.startClusteringKPrototypesMinData()

file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)