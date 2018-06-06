from src.Transformator import Transformator
from src.Classifying.NaiveBayes import NaiveBayes
from src.Classifying.DecisionTree import DecisionTree
from src.Classifying.NeuralNetwork import NeuralNetwork
from src.Classifying.SupportVectorMachine import SupportVectorMachine
from src.Classifying.KNearest import KNearest
from collections import OrderedDict

import matplotlib.pyplot as plt

class Runner:

    def __init__(self, dataset):

        self.dataset = dataset.copy(deep=True)

    def runClassifiers(self):
        self.names = list(OrderedDict.fromkeys(self.dataset['CATEGORIA'].values))
        self.names.pop(0)

        # Transform data
        transformator = Transformator(self.dataset)
        data, trans = transformator.run()

        cv = 6

        # Bayes Naive
        bayes = NaiveBayes(data, cv)
        bayes_resp = bayes.run()

        # DecisionTree
        decisionTree = DecisionTree(data, trans, cv)
        gini_resp = decisionTree.runGini()
        entropy_resp = decisionTree.runEntropy()

        # NeuralNetwork
        neuralNet = NeuralNetwork(data, trans, cv)
        nn_resp = neuralNet.run()

        # SVM
        svm = SupportVectorMachine(data, trans, cv)
        svm_resp = svm.run()

        # K-Nearest
        neigh = KNearest(data, trans, cv)
        neigh_resp = neigh.run()

        plt.clf()
        self.addPlotROCCurves(bayes_resp, 2, 3, 1, "Bernoulli NB")
        plt.ylabel('True Positive Rate')
        self.addPlotROCCurves(gini_resp, 2, 3, 2, "Gini DT")
        self.addPlotROCCurves(entropy_resp, 2, 3, 3, "Entropy DT")
        self.addPlotROCCurves(nn_resp, 2, 3, 4, "Neural net")
        plt.ylabel('True Positive Rate')
        self.addPlotROCCurves(svm_resp, 2, 3, 5, "SVM")
        plt.xlabel('False Positive Rate')
        self.addPlotROCCurves(neigh_resp, 2, 3, 6, "KNN")

        self.showPlotCurves()

        plt.clf()
        self.plotConfusionMatrix(bayes_resp, 2, 3, 1, "Bernoulli NB")
        self.plotConfusionMatrix(gini_resp, 2, 3, 2, "Gini DT")
        self.plotConfusionMatrix(entropy_resp, 2, 3, 3, "Entropy DT")
        self.plotConfusionMatrix(nn_resp, 2, 3, 4, "Neural net")
        self.plotConfusionMatrix(svm_resp, 2, 3, 5, "SVM")
        self.plotConfusionMatrix(neigh_resp, 2, 3, 6, "KNN")

        self.showPlotCurves()