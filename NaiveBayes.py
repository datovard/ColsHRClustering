import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp

from collections import OrderedDict

from Classifier import Classifier

class NaiveBayes(Classifier):
    def __init__(self, dataset):
        self.data = dataset.copy(deep=True)

    def run(self):
        erase_vars = ['FECHA INICIO POSESION']
        self.data.drop(erase_vars, axis=1, inplace=True)

        names = list(OrderedDict.fromkeys(self.data['CATEGORIA'].values))
        y = self.data['CATEGORIA'].astype("category").cat.codes.values
        self.data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1, inplace=True)

        self.data = pd.get_dummies(self.data)

        X = self.data.values

        classifier = OneVsRestClassifier(BernoulliNB())

        ROC = self.getROC( X, y, classifier, len(names) )
        ROC["prediction"] = self.getPrediction( X, y, classifier )
        ROC["y_true"] = y
        return ROC
