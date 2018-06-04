import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize
from Clustering import Cluster
from Classifying import Classify
from Pca import Pca
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from collections import OrderedDict

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

        self.names = list(OrderedDict.fromkeys(self.dataset['CATEGORIA'].values))

        #Discretize
        discretize = Discretize( self.dataset, False )
        self.dataset = discretize.discretizeFile()

        classify = Classify(self.dataset)
        bernoulli = classify.classifyBernoulliNB()

        self.plotROCCurves(bernoulli)
        self.plotConfusionMatrix(bernoulli)

        # cluster1 = Cluster(self.dataset)
        # cluster1.startClusteringKMeans()

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
