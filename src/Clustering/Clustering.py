import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, euclidean
from kmodes.kprototypes import KPrototypes


class Cluster:
    def __init__(self):
        self.startPlotCluster()

    def startPlotCluster(self):
        self.fig = plt.figure(1)

    def plotCluster(self, data, labels, keys, title, subplot ):
        ax = self.fig.add_subplot(subplot[0], subplot[1], subplot[2], projection="3d") #Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels.astype(np.float), edgecolor='k', s=50 )

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(keys[0], labelpad=-10).set_size(7)
        ax.set_ylabel(keys[1], labelpad=-10).set_size(7)
        ax.set_zlabel(keys[2], labelpad=-10).set_size(7)
        ax.set_title(title)
        ax.dist = 12

    def getRemovedDataset(self):
        data = self.dataset.copy(deep=True)
        erase_vars = [ 'TURNO', 'HORAS AL MES', 'HORARIO TRABAJO',
                      'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO',
                      'AREA DE NOMINA', 'CENTRO DE COSTE',
                      'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO',
                      'ROL DEL EMPLEADO', 'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR']

        data = self.dataset.drop(erase_vars, axis=1)
        data = data[[ 'EDAD DEL EMPLEADO', 'SALARIOS MINIMOS', 'AFILIADO A PAC', 'CATEGORIA ESPECIFICA', 'CATEGORIA']]

        return data

    def showPlot(self):
        plt.tight_layout()
        plt.show()
        plt.clf()