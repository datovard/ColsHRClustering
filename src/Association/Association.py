import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from test import Apriori
import pyfpgrowth
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.associations import Associator

class Association:

    def __init__(self, dataset):

        self.dataset = dataset
        jvm.start()

    def apriori(self, keys, num_rules=10, lower_min_support=0.4, upper_min_support=0.9, confidence=0.9):

        dataset = self.dataset.copy(deep=True)
        erase_vars = []
        for k in dataset.keys():
            if k not in keys:
                erase_vars.append(k)
        dataset.drop(erase_vars, axis=1, inplace=True)
        dataset.to_csv('files/weka.csv', sep='\t', index=False)

        data_dir = "files/"
        loader = Loader(classname="weka.core.converters.CSVLoader", options=["-F", "\t", "-N", "first-last"])
        data = loader.load_file(data_dir + "weka.csv")
        data.class_is_last()

        associator = Associator(
            classname="weka.associations.Apriori",
            options=["-N", str(num_rules), "-T", "0", "-C", str(confidence), "-D", "0.05", "-U", str(upper_min_support), "-M", str(lower_min_support),
                     "-S", "-1.0", "-c", "-1"]
        )
        associator.build_associations(data)
        print(associator)

    def filteredApriori(self, keys, num_rules=10, lower_min_support=0.4, upper_min_support=0.9, confidence=0.9):

        dataset = self.dataset.copy(deep=True)
        erase_vars = []
        for k in dataset.keys():
            if k not in keys:
                erase_vars.append(k)
        dataset.drop(erase_vars, axis=1, inplace=True)
        dataset.to_csv('files/weka.csv', sep='\t', index=False)

        data_dir = "files/"
        loader = Loader(classname="weka.core.converters.CSVLoader", options=["-F", "\t", "-N", "first-last"])
        data = loader.load_file(data_dir + "weka.csv")
        data.class_is_last()

        associator = Associator(
            classname="weka.associations.FilteredAssociator",
            options=[
                "-F", "weka.filters.AllFilter ",
                "-c", "-1",
                "-W", "weka.associations.Apriori", "-N", str(num_rules), "-T", "0", "-C", str(confidence), "-D", "0.05", "-U", str(upper_min_support), "-M", str(lower_min_support),
                "-S", "-1.0", "-c", "-1"
            ]
        )
        associator.build_associations(data)
        print(associator)

