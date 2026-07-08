import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

#Creamos una clase que se encargue del preprocesamiento de los datos, esto incluye las transformaciones de las variables y la estandarización de los datos
class DataPreProcessor(BaseEstimator, TransformerMixin):
    """Clase personalizada para preprocesar variables numéricas, categóricas y binarias."""
    
    def __init__(self, columnas_categoricas: list = None, columnas_numericas: list = None, columnas_binarias: list = None) -> None:
        """
        :param columnas_categoricas: lista de las variables categoricas a transformar
        :param columnas_numericas: lista de las variables a estandarizar
        """
        self.columnas_categoricas = columnas_categoricas or []
        self.columnas_numericas = columnas_numericas or []
        self.columnas_binarias = columnas_binarias or  []
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende las transformaciones de las columnas numéricas y categóricas
        :param X: Variables predictoras
        :return self
        """
    
        self.preprocesador = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder( drop='first',handle_unknown='ignore', sparse_output=False), self.columnas_categoricas),
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
        
        # Aplicamos el procesador definido en fit. Esto devuelve un array de NumPy
        X_transformed = self.preprocesador.transform(X)
        
        # Reconstruir el DataFrame con los nuevos nombres de columnas
        return pd.DataFrame(X_transformed, columns=self.preprocesador.get_feature_names_out(),index=X.index)
    
#Creamos una clase que se encargue de mapear la variable objetivo
class TargetEncoder:
    
    def __init__(self) -> None:
        """
        Mapear la variable objetivo. Transforma los valores de la clase: 'good' -  1 y 'bad' - 0
        """
        self.mapping = {'good' : 1, 'bad' : 0}
        
    def transform(self, y : pd.DataFrame) -> pd.DataFrame :
        return y.map(self.mapping)
        
        
    