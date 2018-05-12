import pandas as pd
from datetime import datetime

class Preprocess:
    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessFile(self):

        self.dataset['SALARIOS MINIMOS'] = pd.Series()
        self.dataset['SALARIOS MINIMOS'] = self.dataset['SALARIO A 240']/(737717)

        self.dataset['FECHA INICIO POSESION'] = map( lambda x: datetime.strptime( x, '%d/%m/%Y' ) if x != "00/00/0000" else None, self.dataset['FECHA INICIO POSESION'] )
        self.dataset['FIN'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['FIN'] )
        self.dataset['PRIMERA ALTA'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['PRIMERA ALTA'] )
        self.dataset['FECHA EXPIRACION CONTRATO'] = map( lambda x: datetime.strptime( x, '%d.%m.%Y' ) if x != "31.12.9999" and x != "00.00.0000" else None, self.dataset['FECHA EXPIRACION CONTRATO'] )

        del self.dataset['ID']
        del self.dataset['FECHA DE NACIMIENTO']

        self.dataset = self.dataset.drop( self.dataset[self.dataset["EDAD DEL EMPLEADO"] > 70].index )

        return self.dataset