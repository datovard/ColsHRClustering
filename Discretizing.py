import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

"""
ID
CARGO
FECHA INICIO POSESION
FIN
TURNO
SUELDO TEXTO
SALARIO
HORAS AL MES
HORAS SEMANALES
HORAS DIARIAS
HORARIO TRABAJO
GRUPO PERSONAL
CLASE DE CONTRATO
RELACION LABORAL
TIPO DE PACTO
PRIMERA ALTA
FECHA EXPIRACION CONTRATO
AREA DE NOMINA
CENTRO DE COSTE
DIVISION
DIVISION PERSONAL
SUBDIVISION PERSONAL
AREA DE PERSONAL
SEXO
EDAD DEL EMPLEADO
FECHA DE NACIMIENTO
ROL DEL EMPLEADO
SALARIO A 240
TIPO DE PACTO ESPECIFICO
AFILIADO A PAC
FAMILIAR AFILIADO A PAC
ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR
CATEGORIA ESPECIFICA
CATEGORIA
"""

class Discretize:

    def __init__(self, dataset, flag):
        self.dataset = dataset
        self.flag = flag
        plt.interactive(False)

    def savePlot(self, filename, scale):
        plt.show()
        #plt.tight_layout()
        ##plt.savefig(filename)
        #plt.clf()

    def helper(self, a, b):
        first = ""
        second = ""
        r = []
        for i in xrange(a,b):

            if len(str(i)) == 1:
                first = "0"+str(i)
            else:
                first = str(i)

            if len(str(i+1)) == 1:
                second = "0"+str(i+1)
            else:
                second = str(i+1)

            r.append("("+first+" - "+second+"]")
        return r

    def discretizeFile(self):
        data = self.dataset

        print "DISCRETIZE MODULE \n\n"

        folder = "results/discretizing/"

        # Discretize Age
        print "EDAD DEL EMPLEADO"
        column = data["EDAD DEL EMPLEADO"]
        plt.title("EDAD DEL EMPLEADO")
        grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
        plt.subplot(grid[0,0])
        column.plot(kind='box', title="Diagrama de caja")
        plt.subplot(grid[0,1], title="Normal")
        column.hist(bins=30)
        plt.subplot(grid[0,2])
        column = pd.cut(column, [0,20,30,40,50,60,100], labels=["(... - 20","(20 - 30]","(30 - 40]","(40 - 50]","(50 - 60]","(60 - ...]",])
        column.values.value_counts().plot(kind='bar', title="Discretizado")
        if self.flag: plt.show()
        data["EDAD DEL EMPLEADO"] = column

        # Discretize Salary
        print "SALARIOS MINIMOS"
        column = data["SALARIOS MINIMOS"]
        plt.title("SALARIOS MINIMOS")
        grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
        plt.subplot(grid[0, 0])
        column.plot(kind='box', title="Diagrama de caja")
        plt.subplot(grid[0, 1], title="Normal")
        column.hist(bins=23)
        plt.subplot(grid[0, 2])
        range1 = data[data["SALARIOS MINIMOS"] > 0][data["SALARIOS MINIMOS"] <= 10]
        range2 = data[data["SALARIOS MINIMOS"] > 12][data["SALARIOS MINIMOS"] <= 17]
        range3 = data[data["SALARIOS MINIMOS"] > 20][data["SALARIOS MINIMOS"] <= 21]

        range1["SALARIOS MINIMOS"] = pd.cut(range1["SALARIOS MINIMOS"], range(0,10), labels=self.helper(0,9))
        range2["SALARIOS MINIMOS"] = pd.cut(range2["SALARIOS MINIMOS"], range(12,18), labels=self.helper(12,17))
        range3["SALARIOS MINIMOS"] = pd.cut(range3["SALARIOS MINIMOS"], range(20, 22), labels=self.helper(20,21))

        merged = pd.concat([range1, range2, range3])
        merged["SALARIOS MINIMOS"].value_counts().sort_index().plot(kind='bar', title="SALARIOS MINIMOS")
        if self.flag: plt.show()
        data["SALARIOS MINIMOS"] = merged["SALARIOS MINIMOS"]

        # Discretize Horas al mes
        print "HORAS AL MES"
        column = data["HORAS AL MES"]
        plt.title("HORAS AL MES")
        grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
        plt.subplot(grid[0, 0])
        column.plot(kind='box', title="Diagrama de caja")
        plt.subplot(grid[0, 1], title="Normal")
        column.hist(bins=len(column.value_counts()))
        plt.subplot(grid[0, 2])
        column = pd.cut(column, range(0, 280, 30))#, [0, 20, 30, 40, 50, 60, 100],labels=["(... - 20", "(20 - 30]", "(30 - 40]", "(40 - 50]", "(50 - 60]", "(60 - ...]", ])
        column.values.value_counts().plot(kind='bar', title="Discretizado")
        if self.flag: plt.show()

        grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)

        # Discretize ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR
        #print "AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"
        bins = 7
        column = data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"]
        plt.title("ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR")
        plt.subplot(grid[0, 0])
        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5", "6"])
        column.values.value_counts().plot(kind='bar', title="ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR")
        data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"] = column

        # Discretize AFILIADO A PAC
        #print "AFILIADO A PAC"
        bins = 2
        column = data["AFILIADO A PAC"]
        plt.title("AFILIADO A PAC")
        plt.subplot(grid[0, 1])
        column = pd.cut(column, bins, labels=["0", "1"])
        column.values.value_counts().plot(kind='bar', title="AFILIADO A PAC")
        data["AFILIADO A PAC"] = column

        # Discretize FAMILIAR AFILIADO A PAC
        #print "FAMILIAR AFILIADO A PAC"
        bins = 6
        column = data["FAMILIAR AFILIADO A PAC"]
        plt.title("FAMILIAR AFILIADO A PAC")
        plt.subplot(grid[0, 2])
        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5"])
        column.values.value_counts().plot(kind='bar', title="FAMILIAR AFILIADO A PAC")
        if self.flag: plt.show()
        data["FAMILIAR AFILIADO A PAC"] = column

        # Discretize Initial Date
        column = data["FECHA INICIO POSESION"]
        bins_dt = pd.date_range('2005-01-01', freq='1AS', periods=15)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64) // 10 ** 9, bins=bins_dt.astype(np.int64) // 10 ** 9, labels=labels)
        column.values.value_counts().plot(kind='bar', title="FECHA INICIO POSESION", figsize=(8, 6))
        #if self.flag: self.savePlot(folder + "FECHA INICIO POSESION.png", 0.4)
        data["FECHA INICIO POSESION"] = column

        # Discretize Primera Alta
        column = data["PRIMERA ALTA"]
        bins_dt = pd.date_range('1976-09-28', freq='7AS', periods=7)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64) // 10 ** 9, bins=bins_dt.astype(np.int64) // 10 ** 9, labels=labels)
        column.values.value_counts().plot(kind='bar', title="PRIMERA ALTA")
        #if self.flag: self.savePlot(folder + "PRIMERA ALTA.png", None)
        data["PRIMERA ALTA"] = column


        #print data

        #Renew dataset and return it
        self.dataset = data

        return self.dataset