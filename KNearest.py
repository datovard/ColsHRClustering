from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

class KNearest:

    def __init__(self, X_train, X_test, Y_train, Y_test, trans):

        self.X_train = X_train.copy(deep=True)
        self.X_test = X_test.copy(deep=True)
        self.Y_train = Y_train.copy(deep=True)
        self.Y_test = Y_test.copy(deep=True)
        self.Y_pred = []
        self.trans = trans

    def run(self):

        print "K-NEAREST NEIGHTBOOR"

        #K-NEAREST
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(self.X_train, self.Y_train)
        self.Y_pred = neigh.predict(self.X_test)
        print "Accuracy for K-Nearest Neighboor is:", accuracy_score(self.Y_test, self.Y_pred) * 100
        print(confusion_matrix(self.Y_test, self.Y_pred))
        print(classification_report(self.Y_test, self.Y_pred))