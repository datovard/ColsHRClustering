import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import confusion_matrix, accuracy_score, auc

from src.Transformator import Transformator
from src.Classifying.NaiveBayes import NaiveBayes
from src.Classifying.DecisionTree import DecisionTree
from src.Classifying.NeuralNetwork import NeuralNetwork
from src.Classifying.SupportVectorMachine import SupportVectorMachine
from src.Classifying.KNearest import KNearest
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import random

from sklearn.externals.six import StringIO
from IPython.display import display, Image
from sklearn.tree import export_graphviz
import pydotplus



class Runner:

    def __init__(self, dataset):
        self.dataset = dataset.copy(deep=True)

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
        plt.tight_layout()
        plt.show()
        plt.clf()

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

        self.plotConfusionMatrix(bayes_resp, 2, 3, 1, "Bernoulli NB")
        self.plotConfusionMatrix(gini_resp, 2, 3, 2, "Gini DT")
        self.plotConfusionMatrix(entropy_resp, 2, 3, 3, "Entropy DT")
        self.plotConfusionMatrix(nn_resp, 2, 3, 4, "Neural net")
        self.plotConfusionMatrix(svm_resp, 2, 3, 5, "SVM")
        self.plotConfusionMatrix(neigh_resp, 2, 3, 6, "KNN")

        self.showPlotCurves()


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
                 label=r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
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
        plt.tight_layout()
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

    def drawGiniTree(self):

        # Deep copy of data
        data = self.dataset.copy(deep=True)

        # DELETE Alto from CATEGORIA
        data = data.drop(data[data["CATEGORIA"] == "Alto"].index)

        # Get the keys
        keys = list(data.keys())

        # Transform to every value to number
        trans = {}
        for key in keys:
            c = 0
            trans[key] = {}
            for i in sorted(list(data[key].value_counts().keys())):
                trans[key][i] = c
                c += 1

        for k in keys:
            data_n = []
            for term in data[k]:
                try:
                    data_n.append(trans[k][term])
                except:
                    print k
                    print sorted(trans[k])
                    print term
            data[k] = data_n

        # Separate the classifiers

        Y = data['CATEGORIA']
        # Y = data['CATEGORIA ESPECIFICA']

        # Drop the classifiers
        data = data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1)
        X = data[:]

        # Split 70 to traing and 30 to test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.randint(0, 100))

        # Gini
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=21, max_depth=5, min_samples_leaf=8)
        clf_gini.fit(X_train, Y_train)
        Y_pred = clf_gini.predict(X_test)

        dot_data = StringIO()
        export_graphviz(clf_gini, out_file=dot_data, filled=True, rounded=True, special_characters=True,
        feature_names=X_train.keys(), class_names=sorted(trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)

    def drawEntropyTree(self):

        # Deep copy of data
        data = self.dataset.copy(deep=True)

        # DELETE Alto from CATEGORIA
        data = data.drop(data[data["CATEGORIA"] == "Alto"].index)

        # Get the keys
        keys = list(data.keys())

        # Transform to every value to number
        trans = {}
        for key in keys:
            c = 0
            trans[key] = {}
            for i in sorted(list(data[key].value_counts().keys())):
                trans[key][i] = c
                c += 1

        for k in keys:
            data_n = []
            for term in data[k]:
                try:
                    data_n.append(trans[k][term])
                except:
                    print k
                    print sorted(trans[k])
                    print term
            data[k] = data_n

        # Separate the classifiers

        Y = data['CATEGORIA']
        # Y = data['CATEGORIA ESPECIFICA']

        # Drop the classifiers
        data = data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1)
        X = data[:]

        # Split 70 to traing and 30 to test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.randint(0, 100))

        # Entropy
        clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=23, max_depth=5, min_samples_leaf=8)
        clf_entropy.fit(X_train, Y_train)
        Y_pred = clf_entropy.predict(X_test)

        dot_data = StringIO()
        export_graphviz(clf_entropy, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                        feature_names=X_train.keys(), class_names=sorted(trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)