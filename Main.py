import pandas as pd
from Preprocessing import Preprocess
from Discretizing import Discretize

class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")
        self.start()

    def start(self):

        #Preprocessing
        preprocess = Preprocess( self.dataset )
        self.dataset = preprocess.preprocessFile()

        #Discretize
        discretize = Discretize( self.dataset )
        self.dataset = discretize.discretizeFile()

        """
        np_array = self.dataset.values
        names = list( self.dataset.dtypes.keys())
        record = np_array[0]

        for i in xrange(len(names)):
            print names[i] + ":", record[i], type(record[i])
        """


file = 'files/database.csv'

if __name__ == '__main__':
    run = Main(file)