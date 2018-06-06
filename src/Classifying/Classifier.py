import numpy as np
import seaborn as sns; sns.set()
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import interp

class Classifier:
    def __init__(self, _cv):
        self.cv = _cv

    def getConfusionMatrix(self, classes, y_true, y_pred):
        a = np.zeros(shape=(len(classes), len(classes)), dtype=np.int8)

        matrix = []
        for i in xrange(len(classes)):
            row = []
            for j in xrange(len(classes)):
                row.append(0)
            matrix.append(row)

        for i in xrange(len(y_true)):
            matrix[y_pred[i]][y_true[i]] += 1

        for i in xrange(len(classes)):
            print "   ",i, "\t",
        print
        i = 0
        for row in matrix:
            for j in row:
                if j <= 9:
                    print "   ",
                elif j <= 99:
                    print "  ",
                else:
                    print " ",
                print j, "\t",
            print "|", i,"=",classes[i]
            i += 1

    def getROCValues(self, n_classes, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return  fpr["macro"], tpr["macro"], roc_auc["macro"]

    def getROC(self, X, y, classifier, clasLen):
        # Binarize the output
        y_ = label_binarize(y, classes=xrange(clasLen))
        n_classes = y_.shape[1]

        # Learn to predict each class against the other
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=self.cv)

        total = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train, test in cv.split(X, y):
            classifier.fit(X[train], y_[train])
            probas_ = classifier.predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, roc_auc = self.getROCValues(n_classes, y_[test], probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            total.append([fpr, tpr])

        return {"total": total, "tprs": tprs, "aucs": aucs, "mean_fpr": mean_fpr}

    def getPrediction(self, X, y, classifier):
        prediction = cross_val_predict( classifier, X, y, cv=self.cv )
        return prediction