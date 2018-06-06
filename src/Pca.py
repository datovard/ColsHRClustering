import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder

class Pca:

    def __init__(self, dataset):
        self.dataset = dataset

    def pca_process(self):

        data = self.dataset

        enc = OneHotEncoder()
        X = enc.fit_transform(data).toarray()
        print(X)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        print(X_pca)

