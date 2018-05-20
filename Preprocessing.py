import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta

class Preprocess:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessFile(self):
        print "PREPROCESSING MODULE"

        self.dataset['SALARIOS MINIMOS'] = pd.Series()
        self.dataset['SALARIOS MINIMOS'] = map( lambda x: round(x, 2), self.dataset['SALARIO A 240']/(737717))

        #Missing data in FECHA INICIO POSESION
        mean = datetime.strptime("2005-01-18", '%Y-%m-%d')
        self.dataset['FECHA INICIO POSESION'] = map(lambda x: datetime.strptime(x, '%d/%m/%Y') if x != "00/00/0000" else mean, self.dataset['FECHA INICIO POSESION'])

        mean = datetime.strptime("1732-03-18", '%Y-%m-%d')
        self.dataset['FIN'] = map(lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else mean, self.dataset['FIN'])

        mean = datetime.strptime("2011-04-23", '%Y-%m-%d')
        self.dataset['PRIMERA ALTA'] = map(lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else mean, self.dataset['PRIMERA ALTA'])

        mean = datetime.strptime("1793-04-18", '%Y-%m-%d')
        self.dataset['FECHA EXPIRACION CONTRATO'] = map(lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else mean, self.dataset['FECHA EXPIRACION CONTRATO'])

        """
        #erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'CARGO', 'FECHA INICIO POSESION', 'FIN', 'TURNO', 'SUELDO TEXTO', 'SALARIO', 'HORAS AL MES', 'HORAS SEMANALES', 'HORAS DIARIAS', 'HORARIO TRABAJO', 'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO', 'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA', 'CENTRO DE COSTE', 'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO', 'EDAD DEL EMPLEADO', 'ROL DEL EMPLEADO', 'SALARIO A 240', 'TIPO DE PACTO ESPECIFICO', 'AFILIADO A PAC', 'FAMILIAR AFILIADO A PAC', 'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR', 'CATEGORIA ESPECIFICA', 'CATEGORIA', 'SALARIOS MINIMOS']
        #erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'CARGO', 'FECHA INICIO POSESION', 'FIN', 'TURNO', 'SUELDO TEXTO',
                      'SALARIO', 'HORAS AL MES', 'HORAS SEMANALES', 'HORAS DIARIAS', 'HORARIO TRABAJO',
                      'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO', 'PRIMERA ALTA',
                      'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA', 'CENTRO DE COSTE', 'DIVISION', 'DIVISION PERSONAL',
                      'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO', 'ROL DEL EMPLEADO',
                      'SALARIO A 240', 'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR', 'CATEGORIA ESPECIFICA']
        #self.dataset = self.dataset[['SALARIOS MINIMOS', 'EDAD DEL EMPLEADO', 'AFILIADO A PAC', 'CATEGORIA']]
        """

        erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'SALARIO', 'SUELDO TEXTO', 'HORAS SEMANALES', 'HORAS DIARIAS', 'FIN']
        self.dataset.drop( erase_vars, axis=1, inplace=True )

        self.dataset = self.dataset.drop( self.dataset[self.dataset["EDAD DEL EMPLEADO"] > 70].index )

        return self.dataset