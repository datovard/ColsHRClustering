import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import OrderedDict

from Classifier import Classifier

class NaiveBayes(Classifier):
    def __init__(self, _data, _cv):
        Classifier.__init__(self, _cv)
        self.data = _data.copy(deep=True)
        self.cv = _cv

    def run(self):
        print "BERNOULLI NAIVE BAYES"

        erase_vars = ['FECHA INICIO POSESION']
        self.data.drop(erase_vars, axis=1, inplace=True)

        names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        names.pop(0)
        y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        self.data = pd.get_dummies(self.data)

        X = self.data.values

        classifier = OneVsRestClassifier(BernoulliNB())

        ROC = self.getROC( X, y, classifier, len(names) )
        ROC["prediction"] = self.getPrediction( X, y, classifier )
        ROC["y_true"] = y

        print "Accuracy for Gini is:", accuracy_score(y, ROC["prediction"]) * 100
        print(confusion_matrix(y, ROC["prediction"]))
        print(classification_report(y, ROC["prediction"]))

        return ROC
