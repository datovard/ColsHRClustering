import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp

from itertools import cycle
from collections import OrderedDict


class Classify:
    def __init__(self, dataset):
        self.dataset = dataset

    def getRemovedDataset(self):
        data = self.dataset.copy(deep=True)

        erase_vars = ['CARGO', 'FECHA INICIO POSESION', 'TURNO', 'HORAS AL MES', 'HORARIO TRABAJO',
                      'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO',
                      'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA', 'CENTRO DE COSTE',
                      'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO',
                      'ROL DEL EMPLEADO', 'SALARIO A 240', 'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR']

        data.drop(erase_vars, axis=1, inplace=True)
        data = data[['SALARIOS MINIMOS', 'EDAD DEL EMPLEADO', 'AFILIADO A PAC', 'CATEGORIA ESPECIFICA', 'CATEGORIA']]

        return data

    def getConfusionMatrix(self, classes, y_true, y_pred):
        a = np.zeros(shape=(len(classes), len(classes)), dtype=np.int8)

        matrix = []
        for i in xrange(len(classes)):
            row = []
            for j in xrange(len(classes)):
                row.append(0)
            matrix.append(row)

        for i in xrange(len(y_true)):
            matrix[y_pred[i]][y_true[i]] += 1

        for i in xrange(len(classes)):
            print "   ",i, "\t",
        print
        i = 0
        for row in matrix:
            for j in row:
                if j <= 9:
                    print "   ",
                elif j <= 99:
                    print "  ",
                else:
                    print " ",
                print j, "\t",
            print "|", i,"=",classes[i]
            i += 1

    def getROCValues(self, n_classes, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return  fpr["macro"], tpr["macro"], roc_auc["macro"]

    def classifyBernoulliNB(self):
        data = self.dataset.copy(deep=True)

        erase_vars = ['CARGO', 'FECHA INICIO POSESION', 'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO']
        data.drop(erase_vars, axis=1, inplace=True)

        names = list(OrderedDict.fromkeys(data['CATEGORIA'].values))
        y = data['CATEGORIA'].astype("category").cat.codes.values
        data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        data = pd.get_dummies(data)

        X = data.values

        # Binarize the output
        y_ = label_binarize(y, classes=xrange(len(names)))
        n_classes = y_.shape[1]

        # Learn to predict each class against the other
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=6)
        classifier = OneVsRestClassifier(BernoulliNB())

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        plt.clf()
        for train, test in cv.split(X, y):
            classifier.fit(X[train], y_[train])
            probas_ = classifier.predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, roc_auc = self.getROCValues(n_classes, y_[test], probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
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

        '''mat = confusion_matrix(y_true[0], predictions[0])
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=names, yticklabels=names)
        plt.xlabel('Verdaderos')
        plt.ylabel('Predecidos');
        plt.show()'''













        # Compute ROC curve and ROC area for each class
        #fpr, tpr, roc_auc = self.getROCValues(n_classes, y_test, y_score)

        '''lw = 2
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr, tpr,
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()'''

        '''
        predict = cross_val_predict( bnb, X, Y, cv=10 )


        print "Precision:", accuracy_score(Y, predict) * 100

        mat = confusion_matrix(Y, predict)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=names, yticklabels=names)
        plt.xlabel('Verdaderos')
        plt.ylabel('Predecidos');
        plt.show()

        self.getConfusionMatrix(names, Y, predict)'''