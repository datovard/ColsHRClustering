import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta

class Preprocess:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessFile(self):
        print "PREPROCESSING MODULE \n\n"

        self.dataset = self.dataset.drop(self.dataset[self.dataset["EDAD DEL EMPLEADO"] > 70].index)

        print "Finding Minimum Wages"
        self.dataset['SALARIOS MINIMOS'] = pd.Series()
        self.dataset['SALARIOS MINIMOS'] = map( lambda x: round(x, 2), self.dataset['SALARIO A 240']/(737717))

        #Missing data in FECHA INICIO POSESION
        print "Replacing missing data in FECHA INICIO POSESION"
        mean = datetime.strptime("2005-01-18", '%Y-%m-%d')
        self.dataset['FECHA INICIO POSESION'] = map(lambda x: datetime.strptime(x, '%d/%m/%Y') if x != "00/00/0000" and x != "1/01/1960" else mean, self.dataset['FECHA INICIO POSESION'])

        # Missing data in PRIMERA ALTA
        print "Replacing missing data in PRIMERA ALTA"
        mean = datetime.strptime("2011-04-23", '%Y-%m-%d')
        self.dataset['PRIMERA ALTA'] = map(lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else mean, self.dataset['PRIMERA ALTA'])

        # Missing data in FECHA EXPIRACION CONTRATO
        print "Replacing missing data in FECHA EXPIRACION CONTRATO"
        mean = datetime.strptime("1793-04-18", '%Y-%m-%d')
        self.dataset['FECHA EXPIRACION CONTRATO'] = map(lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else mean, self.dataset['FECHA EXPIRACION CONTRATO'])

        print "Replacing missing data in TURNO"
        self.dataset['TURNO'].fillna(self.dataset['TURNO'].mode()[0], inplace=True)

        print "Replacing missing data in HORARIO TRABAJO"
        self.dataset['HORARIO TRABAJO'].fillna(self.dataset['HORARIO TRABAJO'].mode()[0], inplace=True)

        print "Replacing missing data in CENTRO DE COSTE"
        self.dataset['CENTRO DE COSTE'].fillna(self.dataset['CENTRO DE COSTE'].mode()[0], inplace=True)

        print "Replacing missing data in TIPO DE PACTO ESPECIFICO"
        self.dataset['TIPO DE PACTO ESPECIFICO'].fillna(self.dataset['TIPO DE PACTO ESPECIFICO'].mode()[0], inplace=True)

        #TURNO
        #print "--------------------------------------"
        #print self.dataset["TURNO"].mode()[0]
        #print "--------------------------------------"

        """
        #erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'CARGO', 'FECHA INICIO POSESION', 'FIN', 'TURNO', 'SUELDO TEXTO',
                      'SALARIO', 'HORAS AL MES', 'HORAS SEMANALES', 'HORAS DIARIAS', 'HORARIO TRABAJO',
                      'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL', 'TIPO DE PACTO', 'PRIMERA ALTA',
                      'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA', 'CENTRO DE COSTE', 'DIVISION', 'DIVISION PERSONAL',
                      'SUBDIVISION PERSONAL', 'AREA DE PERSONAL', 'SEXO', 'ROL DEL EMPLEADO',
                      'SALARIO A 240', 'TIPO DE PACTO ESPECIFICO', 'FAMILIAR AFILIADO A PAC',
                      'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR', 'CATEGORIA ESPECIFICA']
        """

        erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'SALARIO', 'SUELDO TEXTO', 'HORAS SEMANALES', 'HORAS DIARIAS', 'FIN', 'SALARIO A 240', 'PRIMERA ALTA','CARGO', 'FECHA EXPIRACION CONTRATO']#, 'TURNO','HORARIO TRABAJO']
        print "Deleting variables tat we consider useless", erase_vars
        self.dataset.drop( erase_vars, axis=1, inplace=True )

        print "La longitud del conjunto de datos es de", self.dataset.shape[0], "filas y", self.dataset.shape[1], "columnas."

        return self.dataset