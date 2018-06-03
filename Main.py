import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster

from Transformator import Transformator
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from NeuralNetwork import NeuralNetwork
from SupportVectorMachine import SupportVectorMachine
from KNearest import KNearest

class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        #Transform data
        transform = Transformator( self.dataset )
        X_train, X_test, Y_train, Y_test, trans = transform.run()

        #Bayes Naive
        bayes = NaiveBayes( X_train, X_test, Y_train, Y_test, trans )
        bayes.run()

        #DecisionTree
        decisionTree = DecisionTree( X_train, X_test, Y_train, Y_test, trans )
        decisionTree.run()

        #NeuralNetwork
        neuralNet = NeuralNetwork( X_train, X_test, Y_train, Y_test, trans )
        neuralNet.run()

        #SVM
        svm = SupportVectorMachine( X_train, X_test, Y_train, Y_test, trans )
        svm.run()

        #K-Nearest
        neigh = KNearest( X_train, X_test, Y_train, Y_test, trans )
        neigh.run()

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