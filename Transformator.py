from sklearn.cross_validation import train_test_split
import random

class Transformator:

    def __init__(self, dataset):

        self.dataset = dataset
        self.trans = {}

    def run(self):
        #Deep copy of data
        data = self.dataset.copy(deep=True)

        #DELETE Alto from CATEGORIA
        data = data.drop(self.dataset[self.dataset["CATEGORIA"] == "Alto"].index)

        #Get the keys
        keys = list(data.keys())

        # Transform to every value to number
        trans = {}
        for key in keys:
            c = 0
            trans[key] = {}
            for i in sorted(list(data[key].value_counts().keys())):
                trans[key][i] = c
                c += 1

        for k in keys:
            data_n = []
            for term in data[k]:
                try:
                    data_n.append(trans[k][term])
                except:
                    print k
                    print sorted(trans[k])
                    print term
            data[k] = data_n

        # Separate the classifiers
        return data, trans

        '''Y = data['CATEGORIA']
        # Y = data['CATEGORIA ESPECIFICA']

        # Drop the classifiers
        data = data.drop(['CATEGORIA ESPECIFICA', 'CATEGORIA'], axis=1)
        X = data[:]

        # Split 70 to traing and 30 to test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.randint(0,100))

        return X_train, X_test, Y_train, Y_test, trans'''
