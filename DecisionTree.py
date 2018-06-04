from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import StringIO
from IPython.display import display, Image
from sklearn.tree import export_graphviz
import pydotplus
from Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier

from collections import OrderedDict

class DecisionTree(Classifier):
    def __init__(self, _data, _trans, _cv):
        Classifier.__init__( self, _cv )
        self.data = _data.copy(deep=True)
        self.trans = _trans

        erase_vars = ['FECHA INICIO POSESION']
        self.data.drop(erase_vars, axis=1, inplace=True)

        self.names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        self.y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        self.X = self.data.values

    def runGini(self):
        print "DECISION TREE"

        #Gini
        clf_gini = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini", random_state=21, max_depth=5, min_samples_leaf=8))
        ROC = self.getROC( self.X, self.y, clf_gini, len(self.names) )
        Y_pred = self.getPrediction( self.X, self.y, clf_gini )

        ROC["prediction"] = Y_pred
        ROC["y_true"] = self.y

        print "Accuracy for Gini is:", accuracy_score(self.y, Y_pred) * 100
        print(confusion_matrix(self.y, Y_pred))
        print(classification_report(self.y, Y_pred))

        return ROC

        '''dot_data = StringIO()
        export_graphviz(clf_gini, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=names, class_names=sorted(self.trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)'''

    def runEntropy(self):
        # Entropy
        clf_entropy = OneVsRestClassifier(DecisionTreeClassifier(criterion="entropy", random_state=23, max_depth=5, min_samples_leaf=8))
        ROC = self.getROC(self.X, self.y, clf_entropy, len(self.names))
        Y_pred = self.getPrediction(self.X, self.y, clf_entropy)

        ROC["prediction"] = Y_pred
        ROC["y_true"] = self.y

        print "Accuracy for Entropy is:", accuracy_score(self.y, Y_pred) * 100
        print(confusion_matrix(self.y, Y_pred))
        print(classification_report(self.y, Y_pred))

        return ROC

        '''dot_data = StringIO()
        export_graphviz(clf_entropy, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=self.X_train.keys(), class_names=sorted(self.trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)'''