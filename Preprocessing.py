#########################
# Column id's
#
# 0  - ID
# 1  - CARGO
# 2  - FECHA INICIO POSESION
# 3  - FIN
# 4  - TURNO
# 5  - SUELDO TEXTO
# 6  - SALARIO
# 7  - HORAS AL MES
# 8  - HORAS SEMANALES
# 9  - HORAS DIARIAS
# 10 - HORARIO TRABAJO
# 11 - GRUPO PERSONAL
# 12 - CLASE DE CONTRATO
# 13 - RELACION LABORAL
# 14 - TIPO DE PACTO
# 15 - PRIMERA ALTA
# 16 - FECHA EXPIRACION CONTRATO
# 17 - AREA DE NOMINA
# 18 - CENTRO DE COSTE
# 19 - DIVISION
# 20 - DIVISION PERSONAL
# 21 - SUBDIVISION PERSONAL
# 22 - AREA DE PERSONAL
# 23 - SEXO
# 24 - EDAD DEL EMPLEADO
# 25 - FECHA DE NACIMIENTO
# 26 - ROL DEL EMPLEADO
# 27 - SALARIO A 240
# 28 - TIPO DE PACTO
# 29 - AFILIADO A PAC
# 30 - FAMILIAR AFILIADO A PAC
# 31 - ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR
# 32 - CATEGORIA ESPECIFICA
# 33 - CATEGORIA
#########################

import numpy as np
import pandas as pd

dataset = pd.read_csv('files/database.csv', index_col=False, header=0, delimiter="\t");

print dataset.keys()
print dataset['SALARIO A 240']

"""
dataset['SALARIO'] = map( lambda x: str.replace( x, ',', '' ), dataset['SALARIO'] )
dataset['SALARIO'] = map( int, dataset['SALARIO'] )

dataset['SALARIO A 240'] = map( lambda x: str.replace( x, ',', '' ), dataset['SALARIO A 240'] )
dataset['SALARIO A 240'] = map( int, dataset['SALARIO A 240'] )

dataset['SALARIOS MINIMOS'] = pd.Series()
dataset['SALARIOS MINIMOS'] = dataset['SALARIO A 240']/(737717)

del dataset['ID']

np_array = dataset.values
names = list(dataset.dtypes.keys())
record = np_array[0]

for i in xrange( len( names ) ):
    print names[i]+":", record[i], type(record[i])

"""