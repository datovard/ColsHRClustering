import numpy as np
import pandas as pd
from datetime import datetime
from Preprocessing import Preprocess

class Main:
    def __init__(self):
        self.dataset = None
        self.preprocessing = Preprocess( 'files/database.csv', self.dataset )

    def start(self):
        self.dataset = self.preprocessing.preprocessFile()

        np_array = self.dataset.values
        names = list( self.dataset.dtypes.keys())
        record = np_array[0]

        for i in xrange(len(names)):
            print names[i] + ":", record[i], type(record[i])

if __name__ == '__main__':
    run = Main()
    run.start()