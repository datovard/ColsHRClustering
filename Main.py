import numpy as np
import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster
#from Classifier import Classify
from Pca import Pca

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from collections import OrderedDict

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

    def addPlotROCCurves(self, data, nrows, ncols, index, title):
        total = data["total"]
        tprs = data["tprs"]
        aucs = data["aucs"]
        mean_fpr = data["mean_fpr"]

        for i in xrange(len(total)):
            fpr = total[i][0]
            tpr = total[i][1]
            roc_auc = aucs[i]
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1

        plt.subplot(nrows, ncols, index)
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
        plt.title(title)
        plt.legend(loc="lower right")
        #plt.show()

    def showPlotCurves(self):
        plt.show()

    def plotConfusionMatrix(self, data, nrows, ncols, index, title):
        y_true = data["y_true"]
        prediction = data["prediction"]

        plt.subplot(nrows, ncols, index)
        mat = confusion_matrix(y_true, prediction)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,
                    xticklabels=map(lambda x: x[0], self.names), yticklabels=map(lambda x: x[0],self.names))
        plt.xlabel('Verdaderos (Accuracy: '+str(accuracy_score(y_true, prediction) * 100)[:6]+'% )')
        plt.ylabel('Predecidos')
        plt.title(title)

    def start(self):
        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        self.runClassifiers()

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

    def runClassifiers(self):
        self.names = list(OrderedDict.fromkeys(self.dataset['CATEGORIA'].values))
        self.names.pop(0)

        # Transform data
        transformator = Transformator(self.dataset)
        data, trans = transformator.run()

        cv = 6

        # Bayes Naive
        bayes = NaiveBayes(data, cv)
        bayes_resp = bayes.run()

        # DecisionTree
        decisionTree = DecisionTree(data, trans, cv)
        gini_resp = decisionTree.runGini()
        entropy_resp = decisionTree.runEntropy()

        # NeuralNetwork
        neuralNet = NeuralNetwork(data, trans, cv)
        nn_resp = neuralNet.run()

        # SVM
        svm = SupportVectorMachine(data, trans, cv)
        svm_resp = svm.run()

        # K-Nearest
        neigh = KNearest(data, trans, cv)
        neigh_resp = neigh.run()

        plt.clf()
        self.addPlotROCCurves(bayes_resp, 2, 3, 1, "Bernoulli NB")
        plt.ylabel('True Positive Rate')
        self.addPlotROCCurves(gini_resp, 2, 3, 2, "Gini DT")
        self.addPlotROCCurves(entropy_resp, 2, 3, 3, "Entropy DT")
        self.addPlotROCCurves(nn_resp, 2, 3, 4, "Neural net")
        plt.ylabel('True Positive Rate')
        self.addPlotROCCurves(svm_resp, 2, 3, 5, "SVM")
        plt.xlabel('False Positive Rate')
        self.addPlotROCCurves(neigh_resp, 2, 3, 6, "KNN")

        self.showPlotCurves()

        plt.clf()
        self.plotConfusionMatrix(bayes_resp, 2, 3, 1, "Bernoulli NB")
        self.plotConfusionMatrix(gini_resp, 2, 3, 2, "Gini DT")
        self.plotConfusionMatrix(entropy_resp, 2, 3, 3, "Entropy DT")
        self.plotConfusionMatrix(nn_resp, 2, 3, 4, "Neural net")
        self.plotConfusionMatrix(svm_resp, 2, 3, 5, "SVM")
        self.plotConfusionMatrix(neigh_resp, 2, 3, 6, "KNN")

        self.showPlotCurves()

file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)
