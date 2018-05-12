import pandas as pd
from Preprocessing import Preprocess


class Main:

    def __init__(self, file):
        self.dataset = pd.read_csv(file, index_col=False, header=0, delimiter="\t")

    def start(self):

        #Preprocessing
        self.preprocessing = Preprocess( self.dataset )
        self.dataset = self.preprocessing.preprocessFile()

        np_array = self.dataset.values
        names = list( self.dataset.dtypes.keys())
        record = np_array[0]

        for i in xrange(len(names)):
            print names[i] + ":", record[i], type(record[i])

file = 'files/database.csv'

if __name__ == '__main__':
    run = Main()
    run.start()