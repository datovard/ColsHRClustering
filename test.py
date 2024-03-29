"""
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Data IO and generation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
"""

from __future__ import division, print_function
import pandas as pd
import numpy as np
import itertools


class Rule():
    def __init__(self, antecedent, concequent, confidence, support):
        self.antecedent = antecedent
        self.concequent = concequent
        self.confidence = confidence
        self.support = support


class Apriori():
    """A method for determining frequent itemsets in a transactional database and
    also for generating rules for those itemsets.

    Parameters:
    -----------
    min_sup: float
        The minimum fraction of transactions an itemets needs to
        occur in to be deemed frequent
    min_conf: float:
        The minimum fraction of times the antecedent needs to imply
        the concequent to justify rule
    """

    def __init__(self, min_sup=0.3, min_conf=0.81):

        self.min_sup = min_sup
        self.min_conf = min_conf
        self.freq_itemsets = None  # List of freqeuent itemsets
        self.transactions = None  # List of transactions

    def _calculate_support(self, itemset):
        count = 0
        for transaction in self.transactions:
            if self._transaction_contains_items(transaction, itemset):
                count += 1
        support = count / len(self.transactions)
        return support

    # Prunes the candidates that are not frequent
    # => returns list with only frequent itemsets
    def _get_frequent_itemsets(self, candidates):
        frequent = []
        # Find frequent items
        for itemset in candidates:
            support = self._calculate_support(itemset)
            if support >= self.min_sup:
                frequent.append(itemset)
        return frequent

    # True or false depending on the candidate has any
    # subset with size k - 1 that is not in the frequent
    # itemset
    def _has_infrequent_itemsets(self, candidate):
        k = len(candidate)
        # Find all combinations of size k-1 in candidate
        # E.g [1,2,3] => [[1,2],[1,3],[2,3]]
        subsets = list(itertools.combinations(candidate, k - 1))
        for t in subsets:
            # t - is tuple. If size == 1 get the element
            subset = list(t) if len(t) > 1 else t[0]
            if not subset in self.freq_itemsets[-1]:
                return True
        return False

    # Joins the elements in the frequent itemset and prunes
    # resulting sets if they contain subsets that have been determined
    # to be infrequent.
    def _generate_candidates(self, freq_itemset):
        candidates = []
        for itemset1 in freq_itemset:
            for itemset2 in freq_itemset:
                # Valid if every element but the last are the same
                # and the last element in itemset1 is smaller than the last
                # in itemset2
                valid = False
                single_item = isinstance(itemset1, int)
                if single_item and itemset1 < itemset2:
                    valid = True
                elif not single_item and np.array_equal(itemset1[:-1], itemset2[:-1]) and itemset1[-1] < itemset2[-1]:
                    valid = True

                if valid:
                    # JOIN: Add the last element in itemset2 to itemset1 to
                    # create a new candidate
                    if single_item:
                        candidate = [itemset1, itemset2]
                    else:
                        candidate = itemset1 + [itemset2[-1]]
                    # PRUNE: Check if any subset of candidate have been determined
                    # to be infrequent
                    infrequent = self._has_infrequent_itemsets(candidate)
                    if not infrequent:
                        candidates.append(candidate)
        return candidates

    # True or false depending on each item in the itemset is
    # in the transaction
    def _transaction_contains_items(self, transaction, items):
        # If items is in fact only one item
        if isinstance(items, int):
            return items in transaction
        # Iterate through list of items and make sure that
        # all items are in the transaction
        for item in items:
            if not item in transaction:
                return False
        return True

    # Returns the set of frequent itemsets in the list of transactions
    def find_frequent_itemsets(self, transactions):
        self.transactions = transactions
        # Get all unique items in the transactions
        unique_items = set(item for transaction in self.transactions for item in transaction)
        # Get the frequent items
        self.freq_itemsets = [self._get_frequent_itemsets(unique_items)]
        while (True):
            # Generate new candidates from last added frequent itemsets
            candidates = self._generate_candidates(self.freq_itemsets[-1])
            # Get the frequent itemsets among those candidates
            frequent_itemsets = self._get_frequent_itemsets(candidates)

            # If there are no frequent itemsets we're done
            if not frequent_itemsets:
                break

            # Add them to the total list of frequent itemsets and start over
            self.freq_itemsets.append(frequent_itemsets)

        # Flatten the array and return every frequent itemset
        frequent_itemsets = [
            itemset for sublist in self.freq_itemsets for itemset in sublist]
        return frequent_itemsets

    # Recursive function which returns the rules where confidence >= min_confidence
    # Starts with large itemset and recursively explores rules for subsets
    def _rules_from_itemset(self, initial_itemset, itemset):
        rules = []
        k = len(itemset)
        # Get all combinations of sub-itemsets of size k - 1 from itemset
        # E.g [1,2,3] => [[1,2],[1,3],[2,3]]
        subsets = list(itertools.combinations(itemset, k - 1))
        support = self._calculate_support(initial_itemset)
        for antecedent in subsets:
            # itertools.combinations returns tuples => convert to list
            antecedent = list(antecedent)
            antecedent_support = self._calculate_support(antecedent)
            # Calculate the confidence as sup(A and B) / sup(B), if antecedent
            # is B in an itemset of A and B
            confidence = float("{0:.2f}".format(support / antecedent_support))
            if confidence >= self.min_conf:
                # The concequent is the initial_itemset except for antecedent
                concequent = [itemset for itemset in initial_itemset if not itemset in antecedent]
                # If single item => get item
                if len(antecedent) == 1:
                    antecedent = antecedent[0]
                if len(concequent) == 1:
                    concequent = concequent[0]
                # Create new rule
                rule = Rule(
                    antecedent=antecedent,
                    concequent=concequent,
                    confidence=confidence,
                    support=support)
                rules.append(rule)

                # If there are subsets that could result in rules
                # recursively add rules from subsets
                if k - 1 > 1:
                    rules += self._rules_from_itemset(initial_itemset, antecedent)
        return rules

    def generate_rules(self, transactions):
        self.transactions = transactions
        frequent_itemsets = self.find_frequent_itemsets(transactions)
        # Only consider itemsets of size >= 2 items
        frequent_itemsets = [itemset for itemset in frequent_itemsets if not isinstance(
            itemset, int)]
        rules = []
        for itemset in frequent_itemsets:
            rules += self._rules_from_itemset(itemset, itemset)
        # Remove empty values
        return rules


def main():
    # Demo transaction set
    # Example 2: https://en.wikipedia.org/wiki/Apriori_algorithm
    transactions = np.array([[1, 2, 3, 4], [1, 2, 4], [1, 2], [2, 3, 4], [2, 3], [3, 4], [2, 4]])
    print("- Apriori -")
    min_sup = 0.25
    min_conf = 0.8
    print("Minimum Support: %.2f, Minimum Confidence: %s" % (min_sup, min_conf))
    print("Transactions:")
    for transaction in transactions:
        print("\t%s" % transaction)

    apriori = Apriori(min_sup=min_sup, min_conf=min_conf)

    # Get and print the frequent itemsets
    frequent_itemsets = apriori.find_frequent_itemsets(transactions)
    print("Frequent Itemsets:\n\t%s" % frequent_itemsets)

    # Get and print the rules
    rules = apriori.generate_rules(transactions)
    print("Rules:")
    for rule in rules:
        print("\t%s -> %s (support: %.2f, confidence: %s)" % (
        rule.antecedent, rule.concequent, rule.support, rule.confidence,))


if __name__ == "__main__":
    main()
