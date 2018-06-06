from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from src.Classifying.Classifier import Classifier
from collections import OrderedDict
from sklearn.multiclass import OneVsRestClassifier

class KNearest(Classifier):

    def __init__(self, _data, _trans, _cv):
        Classifier.__init__(self, _cv)
        self.data = _data.copy(deep=True)
        self.trans = _trans

        self.names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        self.names.pop(0)
        self.y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        self.X = self.data.values

    def run(self):
        print "K-NEAREST NEIGHTBOR"

        #K-NEAREST
        neigh = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
        ROC = self.getROC(self.X, self.y, neigh, len(self.names))
        y_pred = self.getPrediction(self.X, self.y, neigh)

        ROC["prediction"] = y_pred
        ROC["y_true"] = self.y

        print "Accuracy for K-Nearest Neighboor is:", accuracy_score(self.y, y_pred) * 100
        print(confusion_matrix(self.y, y_pred))
        print(classification_report(self.y, y_pred))

        return ROC