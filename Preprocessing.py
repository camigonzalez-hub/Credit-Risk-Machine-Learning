import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

class DataPreProcessor(BaseEstimator, TransformerMixin):
    """Clase personalizada para preprocesar variables numéricas, categóricas y binarias."""
    
    def __init__(self, columnas_categoricas: list = None, categorias_referencia : list = None, columnas_numericas: list = None, columnas_binarias: list = None) -> None:
        """
        :param columnas_categoricas: lista de las variables categoricas a transformar
        :param categorias_referencia: lista de las categorias que se tomarán como referencia para ser eliminadas mediante One Hot Encoder
        :param columnas_numericas: lista de las variables numéricas que se estandarizarán
        :param columnas_binarias: lista de las variables binarias que se mappearan a 0 y 1. 
        """
        self.columnas_categoricas = columnas_categoricas 
        self.categorias_referencia = categorias_referencia 
        self.columnas_numericas = columnas_numericas 
        self.columnas_binarias = columnas_binarias 
    
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende las transformaciones de las columnas numéricas y categóricas
        :param X: Variables predictoras
        :return self
        """
    
        self.preprocesador = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(drop =  self.categorias_referencia ,handle_unknown='error', sparse_output=False), self.columnas_categoricas),
            ('num', StandardScaler(), self.columnas_numericas),
            ('bin', OrdinalEncoder(), self.columnas_binarias)
        ], remainder="passthrough") 
        
        # Ajustamos el preprocesador con los datos originales
        self.preprocesador.fit(X)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones aprendidas a un DataFrame.
        :param X: Variables predictoras
        :return: Pd.DataFrame
        """
        check_is_fitted(self,'preprocesador')

        X_transformed = self.preprocesador.transform(X)

        return pd.DataFrame(X_transformed, columns=self.preprocesador.get_feature_names_out(),index=X.index)
    

class TargetMapper: 
    """ Mapear la variable objetivo. Transforma los valores de la clase: 'good' -  0 y 'bad' - 1 """
    
    def __init__(self) -> None:
        self.mapping = {'good' : 0, 'bad' : 1}
        
    def transform(self, y : pd.Series) -> pd.Series :
        """ Transforma la columna de la variable target """
        return y.map(self.mapping)
        
        
    