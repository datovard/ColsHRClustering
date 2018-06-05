from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict
from Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier

class SupportVectorMachine(Classifier):
    def __init__(self, _data, _trans, _cv):
        Classifier.__init__(self, _cv)
        self.data = _data.copy(deep=True)

        self.names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        self.y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)
        self.X = self.data.values

    def run(self):
        print "SUPPORT VECTOR MACHINE"

        #Standard Support Vector Machine
        clf_svc = OneVsRestClassifier(SVC(probability=True))
        ROC = self.getROC(self.X, self.y, clf_svc, len(self.names))
        Y_pred = self.getPrediction(self.X, self.y, clf_svc)

        ROC["prediction"] = Y_pred
        ROC["y_true"] = self.y

        print "Accuracy for SVM is:", accuracy_score(self.y, Y_pred) * 100
        print(confusion_matrix(self.y, Y_pred))
        print(classification_report(self.y, Y_pred))

        return ROC
