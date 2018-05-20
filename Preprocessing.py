import pandas as pd
from datetime import datetime

class Preprocess:
    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessFile(self):

        print "PREPROCESSING MODULE"

        self.dataset['SALARIOS MINIMOS'] = pd.Series()
        self.dataset['SALARIOS MINIMOS'] = map( lambda x: round(x, 2), self.dataset['SALARIO A 240']/(737717))

        self.dataset['FECHA INICIO POSESION'] = map( lambda x: datetime.strptime( x, '%d/%m/%Y' ) if x != "00/00/0000" else None, self.dataset['FECHA INICIO POSESION'] )
        self.dataset['FIN'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['FIN'] )
        self.dataset['PRIMERA ALTA'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['PRIMERA ALTA'] )
        self.dataset['FECHA EXPIRACION CONTRATO'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['FECHA EXPIRACION CONTRATO'] )

        #erase_vars = ['ID', 'FECHA DE NACIMIENTO', 'CARGO', 'FECHA INICIO POSESION', 'FIN', 'TURNO',
        #              'SUELDO TEXTO', 'SALARIO', 'HORAS AL MES', 'HORAS SEMANALES', 'HORAS DIARIAS',
        #              'HORARIO TRABAJO', 'GRUPO PERSONAL', 'CLASE DE CONTRATO', 'RELACION LABORAL',
        #              'TIPO DE PACTO', 'PRIMERA ALTA', 'FECHA EXPIRACION CONTRATO', 'AREA DE NOMINA',
        #              'CENTRO DE COSTE', 'DIVISION', 'DIVISION PERSONAL', 'SUBDIVISION PERSONAL',
        #              'AREA DE PERSONAL', 'SEXO', 'EDAD DEL EMPLEADO', 'ROL DEL EMPLEADO', 'SALARIO A 240',
        #              'TIPO DE PACTO ESPECIFICO', 'AFILIADO A PAC', 'FAMILIAR AFILIADO A PAC',
        #              'ES AFILIADO A PAC O TIENE AFILIADO A UN FAMILIAR', 'CATEGORIA ESPECIFICA',
        #              'CATEGORIA', 'SALARIOS MINIMOS']

        erase_vars = ['ID', 'FECHA DE NACIMIENTO']

        self.dataset.drop( erase_vars, axis=1, inplace=True )
        self.dataset = self.dataset.drop( self.dataset[self.dataset["EDAD DEL EMPLEADO"] > 70].index )

        return self.dataset