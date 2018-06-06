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

        self.names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        self.names.pop(0)
        self.y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        self.data = pd.get_dummies(self.data)

        self.X = self.data.values

    def run(self):
        print "BERNOULLI NAIVE BAYES"

        clr_nb = OneVsRestClassifier(BernoulliNB())
        ROC = self.getROC( self.X, self.y, clr_nb, len(self.names) )
        y_pred = self.getPrediction( self.X, self.y, clr_nb )

        ROC["prediction"] = y_pred
        ROC["y_true"] = self.y

        print "Accuracy for Bernoulli Naive Bayes is:", accuracy_score(self.y, y_pred) * 100
        print(confusion_matrix(self.y, y_pred))
        print(classification_report(self.y, y_pred))

        return ROC
