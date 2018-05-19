import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

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

        print "DISCRETIZE MODULE"

        data = self.dataset

        """
        #Discretize Age
        bins = 8
        data["EDAD DEL EMPLEADO"] = pd.cut(data["EDAD DEL EMPLEADO"], bins)
        print data["EDAD DEL EMPLEADO"].values.value_counts()
        data["EDAD DEL EMPLEADO"].values.value_counts().plot(kind='bar')
        plt.show()
        """

        print len(data)

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
        merged["SALARIO A 240"].value_counts().sort_index().plot(kind='bar')
        plt.show()

        #Renew dataset and return it
        self.dataset = data
        return self.dataset