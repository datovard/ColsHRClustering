import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing

class DecisionTree:

    def __init__(self, dataset):

        self.dataset = dataset


    def run(self):

        data = self.dataset

        #Index for Tree
        keys = [
            'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO',
            'AREA DE NOMINA', 'DIVISION', 'SEXO', 'EDAD DEL EMPLEADO',
            'ROL DEL EMPLEADO', 'SALARIOS MINIMOS', 'TIPO DE PACTO ESPECIFICO', 'AFILIADO A PAC', 'FAMILIAR AFILIADO A PAC',
            'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR', 'CATEGORIA ESPECIFICA',
        ]

        #Transform to Numbers
        trans = {}
        for k in keys:
            trans[k] = {}
            t_counter = 0
            data_n = []
            for term in data[k]:
                if term not in trans[k]:
                    trans[k][term] = t_counter
                    t_counter +=1
                data_n.append(trans[k][term])
            data[k] = data_n

        for k in data.keys():
            if k not in keys:
                data.drop(k, axis=1, inplace=True)

        #Split Categories and Variables
        X = data.values[:, 0:-2]
        Y = data.values[:, -1]

        #Split 70 to traing and 30 to test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.3, random_state=100)

        #Gini
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        clf_gini.fit(X_train, Y_train)
        Y_pred = clf_gini.predict(X_test)
        print "Accuracy for Gini is:", accuracy_score(Y_test, Y_pred) * 100

        #Entropy
        clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
        clf_entropy.fit(X_train, Y_train)
        Y_pred = clf_entropy.predict(X_test)
        print "Accuracy for Entropy is:", accuracy_score(Y_test, Y_pred) * 100


