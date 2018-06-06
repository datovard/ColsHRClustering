from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from Classifier import Classifier

from collections import OrderedDict

class NeuralNetwork(Classifier):

    def __init__(self, _data, _trans, _cv):
        Classifier.__init__(self, _cv)
        self.data = _data.copy(deep=True)
        self.trans = _trans

        self.names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        self.y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)
        self.X = self.data.values

    def run(self):

        print "MULTILAYER PERCEPTRON NEURAL NETWORK"

        clf_mp = OneVsRestClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (200, 80), random_state = 1))
        ROC = self.getROC(self.X, self.y, clf_mp, len(self.names))
        Y_pred = self.getPrediction(self.X, self.y, clf_mp)

        ROC["prediction"] = Y_pred
        ROC["y_true"] = self.y

        print "Accuracy for MLP is:", accuracy_score(self.y, Y_pred) * 100
        print(confusion_matrix(self.y, Y_pred))
        print(classification_report(self.y, Y_pred))

        return ROC