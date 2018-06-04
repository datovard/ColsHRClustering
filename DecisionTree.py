import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn import preprocessing

from sklearn.externals.six import StringIO
from IPython.display import display, Image
from sklearn.tree import export_graphviz
import pydotplus

class DecisionTree:

    def __init__(self, X_train, X_test, Y_train, Y_test, trans):

        self.X_train = X_train.copy(deep=True)
        self.X_test = X_test.copy(deep=True)
        self.Y_train = Y_train.copy(deep=True)
        self.Y_test = Y_test.copy(deep=True)
        self.Y_pred = []
        self.trans = trans


    def run(self):

        print "DECISION TREE"

        #Gini
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=21, max_depth=5, min_samples_leaf=8)
        clf_gini.fit(self.X_train, self.Y_train)
        self.Y_pred = clf_gini.predict(self.X_test)
        print "Accuracy for Gini is:", accuracy_score(self.Y_test, self.Y_pred) * 100
        print(confusion_matrix(self.Y_test, self.Y_pred))
        print(classification_report(self.Y_test, self.Y_pred))

        dot_data = StringIO()
        export_graphviz(clf_gini, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=self.X_train.keys(), class_names=sorted(self.trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)

        #Entropy
        clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=23, max_depth=5, min_samples_leaf=8)
        clf_entropy.fit(self.X_train, self.Y_train)
        self.Y_pred = clf_entropy.predict(self.X_test)
        print "Accuracy for Entropy is:", accuracy_score(self.Y_test, self.Y_pred) * 100
        print(confusion_matrix(self.Y_test, self.Y_pred))
        print(classification_report(self.Y_test, self.Y_pred))

        dot_data = StringIO()
        export_graphviz(clf_entropy, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=self.X_train.keys(), class_names=sorted(self.trans['CATEGORIA'].keys()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())
        display(img)
