import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster
#from Classifier import Classify
from Pca import Pca
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from Transformator import Transformator
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from NeuralNetwork import NeuralNetwork
from SupportVectorMachine import SupportVectorMachine
from KNearest import KNearest

import random

class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def plotROCCurves(self, data):
        total = data["total"]
        tprs = data["tprs"]
        aucs = data["aucs"]
        mean_fpr = data["mean_fpr"]

        plt.clf()
        for i in xrange(len(total)):
            fpr = total[i][0]
            tpr = total[i][1]
            roc_auc = aucs[i]
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def plotConfusionMatrix(self, data):
        y_true = data["y_true"]
        prediction = data["prediction"]

        mat = confusion_matrix(y_true, prediction)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                    xticklabels=self.names, yticklabels=self.names)
        plt.xlabel('Verdaderos (Accuracy: '+str(accuracy_score(y_true, prediction) * 100)[:6]+'% )')
        plt.ylabel('Predecidos');
        plt.show()

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        self.names = list(OrderedDict.fromkeys(self.dataset['CATEGORIA'].values))
        self.names.pop(0)

        #Transform data
        transformator = Transformator( self.dataset )
        data, trans = transformator.run()

        #Bayes Naive
        bayes = NaiveBayes( data )
        resp = bayes.run()

        self.plotROCCurves(resp)
        self.plotConfusionMatrix(resp)

        Y = data['CATEGORIA']
        # Y = data['CATEGORIA ESPECIFICA']

        # Drop the classifiers
        data = data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1)
        X = data[:]

        # Split 70 to traing and 30 to test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.randint(0, 100))

        #DecisionTree
        decisionTree = DecisionTree( X_train, X_test, Y_train, Y_test, trans )
        decisionTree.run()

        #NeuralNetwork
        '''neuralNet = NeuralNetwork( X_train, X_test, Y_train, Y_test, trans )
        neuralNet.run()

        #SVM
        svm = SupportVectorMachine( X_train, X_test, Y_train, Y_test, trans )
        svm.run()

        #K-Nearest
        neigh = KNearest( X_train, X_test, Y_train, Y_test, trans )
        neigh.run()'''

        #Cluster
        # cluster = Cluster( self.dataset )
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

        #PCA
        # pca = Pca( self.dataset )
        # pca.pca_process()

file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)
