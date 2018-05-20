import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
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
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()

    def discretizeFile(self):
        data = self.dataset

        print "DISCRETIZANDO..."

        folder = "results/discretizing/"

        #Discretize Age
        bins = 8
        column = data["EDAD DEL EMPLEADO"]
        column = pd.cut(column, bins)

        column.values.value_counts().plot(kind='bar', title="EDAD DEL EMPLEADO")
        if self.flag: self.savePlot( folder + "EDAD DEL EMPLEADO.png", None )
        data["EDAD DEL EMPLEADO"] = column

        #Discretize Salary
        bins = 6

        range1 = data[data["SALARIO A 240"] > 0][data["SALARIO A 240"] <= 8714792]
        range2 = data[data["SALARIO A 240"] > 8714792][data["SALARIO A 240"] <= 21757176]

        range1["SALARIO A 240"] = pd.cut(range1["SALARIO A 240"], 8, labels=["A:(13019.37, 876198.75]","B:(876198.75, 1732527.5]","C:(1732527.5, 2588856.25]","D:(2588856.25, 3445185]","E:(3445185, 4301513.75]","F:(4301513.75, 5157842.5]","G:(5157842.5, 6014171.25]","H:(6014171.25, 6870500]"])
        range2["SALARIO A 240"] = pd.cut(range2["SALARIO A 240"], 3, labels=["H:(9578233.224, 13645992]","I:(13645992, 17701584]","J:(17701584, 21757176]"])

        merged = pd.concat([range1, range2])

        merged["SALARIO A 240"].value_counts().sort_index().plot(kind='bar', title="SALARIO A 240")
        data["SALARIO A 240"] = merged["SALARIO A 240"]
        if self.flag: self.savePlot( folder + "SALARIO A 240.png", None )

        # Discretize Initial Date
        column = data["FECHA INICIO POSESION"]
        bins_dt = pd.date_range('2005-01-01', freq='1AS', periods=15)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64)//10**9,bins=bins_dt.astype(np.int64)//10**9,labels=labels)

        column.values.value_counts().plot(kind='bar', title="FECHA INICIO POSESION",figsize=(8,6))
        if self.flag: self.savePlot(folder + "FECHA INICIO POSESION.png", 0.4)
        data["FECHA INICIO POSESION"] = column

        # Discretize Primera Alta
        column = data["PRIMERA ALTA"]

        bins_dt = pd.date_range('1976-09-28', freq='7AS', periods=7)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64) // 10 ** 9, bins=bins_dt.astype(np.int64) // 10 ** 9, labels=labels)

        column.values.value_counts().plot(kind='bar', title="PRIMERA ALTA")
        if self.flag: self.savePlot(folder + "PRIMERA ALTA.png", None)
        data["PRIMERA ALTA"] = column

        # Discretize Horas al mes
        bins = 10
        column = data['HORAS AL MES']
        column = pd.cut(column, bins)

        column.values.value_counts().plot(kind='bar', title='HORAS AL MES')
        if self.flag: self.savePlot(folder + "HORAS AL MES.png", None)
        data['HORAS AL MES'] = column

        # Salarios Minimos
        bins = 10
        column = data["SALARIOS MINIMOS"]
        column = pd.cut(column, bins)

        column.values.value_counts().plot(kind='bar', title="SALARIOS MINIMOS")
        if self.flag: self.savePlot(folder + "SALARIOS MINIMOS.png", None)
        data["SALARIOS MINIMOS"] = column

        # Discretize ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR
        bins = 7
        column = data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"]
        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5", "6"])

        column.values.value_counts().plot(kind='bar', title="ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR")
        if self.flag: self.savePlot(folder + "ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR.png", None)
        data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"] = column

        # Discretize AFILIADO A PAC
        bins = 2
        column = data["AFILIADO A PAC"]

        column = pd.cut(column, bins, labels=["0", "1"])

        column.values.value_counts().plot(kind='bar', title="AFILIADO A PAC")
        if self.flag: self.savePlot(folder + "AFILIADO A PAC.png", None)
        data["AFILIADO A PAC"] = column

        # Discretize AFILIADO A PAC
        bins = 6
        column = data["FAMILIAR AFILIADO A PAC"]

        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5"])

        column.values.value_counts().plot(kind='bar', title="FAMILIAR AFILIADO A PAC")
        if self.flag: self.savePlot(folder + "FAMILIAR AFILIADO A PAC.png", None)
        data["FAMILIAR AFILIADO A PAC"] = column
        #print data

        #Renew dataset and return it
        self.dataset = data
        return self.dataset