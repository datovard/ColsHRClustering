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

    def __init__(self, dataset):
        self.dataset = dataset
        plt.interactive(False)

    def discretizeFile(self):

        data = self.dataset

        print "DISCRETIZE MODULE"
        print data.dtypes
        #print data.head()
        #print data.keys()

        #Discretize Age
        bins = 8
        column = data["EDAD DEL EMPLEADO"]
        column = pd.cut(column, bins)
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="EDAD DEL EMPLEADO")
        plt.show()
        data["EDAD DEL EMPLEADO"] = column

        #Discretize Salary
        bins = 6
        print max(data["SALARIO A 240"])
        range1 = data[data["SALARIO A 240"] > 0][data["SALARIO A 240"] <= 8714792]
        range2 = data[data["SALARIO A 240"] > 8714792][data["SALARIO A 240"] <= 21757176]

        range1["SALARIO A 240"] = pd.cut(range1["SALARIO A 240"], 8, labels=["a","b","c","d","e","f","g","h"])
        range2["SALARIO A 240"] = pd.cut(range2["SALARIO A 240"], 3, labels=["i","j","k"])

        #range1["SALARIO A 240"].value_counts().plot(kind='bar')
        #range2["SALARIO A 240"].value_counts().plot(kind='bar')

        merged = pd.concat([range1, range2])
        print merged["SALARIO A 240"].value_counts().sort_index()
        merged["SALARIO A 240"].value_counts().sort_index().plot(kind='bar', title="SALARIO A 240")
        data["SALARIO A 240"] = merged["SALARIO A 240"]
        plt.show()

        # Discretize Initial Date
        column = data["FECHA INICIO POSESION"]
        bins_dt = pd.date_range('2006-01-01', freq='1AS', periods=9)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64)//10**9,bins=bins_dt.astype(np.int64)//10**9,labels=labels)
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="FECHA INICIO POSESION")
        plt.show()
        data["FECHA INICIO POSESION"] = column

        # Discretize Primera Alta
        column = data["PRIMERA ALTA"]
        print min(column)
        print max(column)
        bins_dt = pd.date_range('1976-09-28', freq='7AS', periods=7)
        bins_str = bins_dt.astype(str).values
        labels = ['({}, {}]'.format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
        column = pd.cut(column.astype(np.int64) // 10 ** 9, bins=bins_dt.astype(np.int64) // 10 ** 9, labels=labels)
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="PRIMERA ALTA")
        plt.show()
        data["PRIMERA ALTA"] = column

        # Discretize Salario
        bins = 10
        column = data['HORAS AL MES']
        column = pd.cut(column, bins)
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title='HORAS AL MES')
        plt.show()
        data['HORAS AL MES'] = column

        # Salarios Minimos
        bins = 10
        column = data["SALARIOS MINIMOS"]
        column = pd.cut(column, bins)
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="SALARIOS MINIMOS")
        plt.show()
        data["SALARIOS MINIMOS"] = column

        # Discretize ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR
        bins = 7
        column = data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"]
        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5", "6"])
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR")
        plt.show()
        data["ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR"] = column

        # Discretize AFILIADO A PAC
        bins = 2
        column = data["AFILIADO A PAC"]
        print "MAX", max(column)
        print "MIN", min(column)
        column = pd.cut(column, bins, labels=["0", "1"])
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="AFILIADO A PAC")
        plt.show()
        data["AFILIADO A PAC"] = column

        # Discretize AFILIADO A PAC
        bins = 6
        column = data["FAMILIAR AFILIADO A PAC"]
        print "MAX", max(column)
        print "MIN", min(column)
        column = pd.cut(column, bins, labels=["0", "1", "2", "3", "4", "5"])
        print column.values.value_counts()
        column.values.value_counts().plot(kind='bar', title="FAMILIAR AFILIADO A PAC")
        plt.show()
        data["FAMILIAR AFILIADO A PAC"] = column

        #print data

        #Renew dataset and return it
        self.dataset = data
        return self.dataset