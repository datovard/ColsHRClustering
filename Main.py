from __future__ import division, print_function
import numpy as np
import pandas as pd
#from Classifier import Classify
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score, confusion_matrix, auc

from src.Preprocessing.Preprocessing import Preprocess
from src.Preprocessing.Discretizing import Discretize
from src.Association.Association import Association
from src.Clustering.Kmodes import Kmodes
from src.Clustering.Kprototypes import Kprototypes
from src.Classifying.Runner import Runner

import weka.core.jvm as jvm


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

        #kmeans = Kmeans(self.dataset)
        #kmeans.run()

        kprototypes = Kprototypes(self.dataset)
        kprototypes.startClusteringKPrototypesHuang()
        kprototypes.startClusteringKPrototypesCao()

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
        kmodes.startClusteringKModesFullDataHuang()
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
