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
plt.interactive(False)

dataset = pd.read_csv('files/database.csv', index_col=False, header=0, delimiter="\t");

for r in dataset[dataset["EDAD DEL EMPLEADO"] > 70].index:
    print dataset.loc[r], "\n-------------------------------------------------------------------------------------------"
dataset = dataset.drop(dataset[dataset["EDAD DEL EMPLEADO"] > 70].index)

#DISCRETIZE AGE
bins = 8
age = dataset["EDAD DEL EMPLEADO"]
age = pd.cut(age, bins)
print age.values.value_counts()
grap = age.values.value_counts()
grap.plot(kind='bar')
plt.show()

