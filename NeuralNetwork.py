from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class NeuralNetwork:

    def __init__(self, X_train, X_test, Y_train, Y_test, trans):

        self.X_train = X_train.copy(deep=True)
        self.X_test = X_test.copy(deep=True)
        self.Y_train = Y_train.copy(deep=True)
        self.Y_test = Y_test.copy(deep=True)
        self.Y_pred = []
        self.trans = trans

    def run(self):

        print "MULTILAYER PERCEPTRON NEURAL NETWORK"

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (200, 80), random_state = 1)
        clf.fit(self.X_train, self.Y_train)
        self.Y_pred = clf.predict(self.X_test)
        print "Accuracy for MLP is:", accuracy_score(self.Y_test, self.Y_pred) * 100
        print(confusion_matrix(self.Y_test, self.Y_pred))
        print(classification_report(self.Y_test, self.Y_pred))