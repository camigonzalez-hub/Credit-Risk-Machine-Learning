import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

class ObtenerProbabilidades:
    """ Obtención de las probabilidades arrojadas por un modelo de regresión logistica respecto a la clase objetivo"""
     
    def __init__(self, pipeline : Pipeline, X_test : pd.DataFrame, y_test : pd.DataFrame) -> None:
         """
         Inicializa la obtención de las probabilidades.
         :param Pipeline: Recibe el pipeline correspondiente al modelo del cual se desean obtener las probabilidades
         :param X_test: Recibe el conjunto de variables explicativas de prueba
         :param y_test: Recibe una columna con la variable target del conjunto de prueba.
         """
         
         self.pipeline = pipeline
         self.X_test = X_test
         self.y_test = y_test
         
    def calcular_probabilidades(self) -> pd.DataFrame:     
        """
        Calcula las probabilidades y las devuelve.
        :return: probabilidades asignadas de pertenencia a cada clase
        """
        check_is_fitted(self.pipeline)
        probabilidades = self.pipeline.predict_proba(self.X_test)
        score = self.pipeline.decision_function(self.X_test)
        predicciones = self.pipeline.predict(self.X_test)

        resultado = pd.DataFrame({
           "Clase real": self.y_test,
            "Probabilidad buen crédito [0]": probabilidades[:, 0],
            "Probabilidad mal crédito [1]": probabilidades[:, 1],
            "Clase predicha": predicciones,
            "Score": score }, index=self.X_test.index) 
        
        return resultado
     
    
class ValidacionCruzada:
    """Realiza una evaluación de Validación Cruzada a nuestro conjunto de entrenamiento"""

    def __init__(self, pipeline : Pipeline, X_train : pd.DataFrame, y_train : pd.Series, n_splits=5, random_state=42) -> None:
        """
        Inicializa la evaluación del modelo mediante validación cruzada.

        :param pipeline: Pipeline del modelo que se desea evaluar.
        :param X_train: Conjunto de entrenamiento de las variables explicativas.
        :param y_train: Variable objetivo correspondiente al conjunto de entrenamiento.
        :param n_splits: Número de particiones utilizadas en la validación cruzada.
        :param random_state: Semilla utilizada para garantizar la reproducibilidad de las particiones.
        """
        
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train

        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        self.scores = None

    def validar(self) -> dict:
        """
        Realiza las validaciones con respecto a las métricas: "roc_auc", "average_precision", "precision", "recall", "f1".
        :return: diccionario con el valor de cada métrica.
        """
        check_is_fitted(self.pipeline)
        self.scores = cross_validate(
            estimator=self.pipeline,
            X=self.X_train,
            y=self.y_train,
            cv=self.cv,
            scoring={
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1"
            },
            return_train_score=False
        )

        return self.scores

    def obtener_resultados(self) -> pd.DataFrame:
        """ Devuelve los resultados de los cálculos en un Data Frame """

        if self.scores is None:
            raise RuntimeError("Aún no se realizó la validación")

        metricas = ["roc_auc","pr_auc","precision","recall","f1"]

        resultados = []

        for metrica in metricas:
            valores = self.scores[f"test_{metrica}"]

            resultados.append({
                "Métrica": metrica,
                "Media": valores.mean(),
                "Desviación Estandar": valores.std()
            })

        return pd.DataFrame(resultados)
    

class CurvaPR:
    """ Crea una Curva PR individual para un modelo dado """
    def __init__(self, pipeline : Pipeline, X_test : pd.DataFrame, y_test : pd.Series) -> None:
        """ 
        Inicializa los parámetros necesarios para el cálculo de la curva.
        :param pipeline: Pipeline del modelo que se desea evaluar.
        :param X_test: Conjunto de prueba de las variables explicativas.
        :param y_test: Variable objetivo correspondiente al conjunto de prueba. 
        """
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        
    def crear_curva(self):
        check_is_fitted(self.pipeline)
        probabilidades = self.pipeline.predict_proba(self.X_test)
        precision,recall, thresholds = precision_recall_curve(self.y_test, probabilidades[:,1])
        pr_auc = auc(recall,precision)
   
        plt.plot(recall,precision, label = f'Presicion - Recall curve = {pr_auc:.4f}')
        plt.xlabel('Recall (sensibilidad)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend()
        plt.show()      

class CurvaRoc:
    """ Crea una Curva ROC individual para un modelo dado """
    def __init__(self, pipeline: Pipeline, X_test : pd.DataFrame, y_test : pd.Series) -> None:
        """ 
        Inicializa los parámetros necesarios para el cálculo de la curva.
        :param pipeline: Pipeline del modelo que se desea evaluar.
        :param X_test: Conjunto de prueba de las variables explicativas.
        :param y_test: Variable objetivo correspondiente al conjunto de prueba. 
        """
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        
    def crear_curva(self) -> None:
        check_is_fitted(self.pipeline)
        probabilidades = self.pipeline.predict_proba(self.X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, probabilidades)
        roc_auc = roc_auc_score(self.y_test, probabilidades)
            
        print(f"ROC AUC: {roc_auc:.3f}")

        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Clasificador aleatorio')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()


class CurvaRocComparaciones:
    """ Crea una curva ROC que compara distintos modelos """
    def __init__( self, modelos: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """ 
        Inicializa los parámetros necesarios para el cálculo de la curva.
        :param modelos: Diccionario que contiene el nombre del modelo y el pipeline del modelo que se desea evaluar.
        :param X_test: Conjunto de prueba de las variables explicativas.
        :param y_test: Variable objetivo correspondiente al conjunto de prueba. 
        """
        self.modelos = modelos
        self.X_test = X_test
        self.y_test = y_test

    def crear_curva(self) -> pd.DataFrame:

        resultados = []

        plt.figure(figsize=(8, 6))

        for nombre, pipeline in self.modelos.items():
            check_is_fitted(pipeline)
            probabilidades = pipeline.predict_proba( self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, probabilidades)
            roc_auc = roc_auc_score( self.y_test, probabilidades)

            resultados.append({
                "Modelo": nombre,
                "ROC AUC": roc_auc})

            plt.plot(fpr, tpr, label=f"{nombre} — AUC = {roc_auc:.4f}")

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Clasificador aleatorio"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Comparación de curvas ROC")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(resultados).sort_values("ROC AUC", ascending= False)
    
class CurvaPRComparaciones:
    """ Crea una curva PR que compara distintos modelos """
    
    def __init__( self, modelos: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """ 
        Inicializa los parámetros necesarios para el cálculo de la curva.
        :param modelos: Diccionario que contiene el nombre del modelo y el pipeline del modelo que se desea evaluar.
        :param X_test: Conjunto de prueba de las variables explicativas.
        :param y_test: Variable objetivo correspondiente al conjunto de prueba. 
        """
        self.modelos = modelos
        self.X_test = X_test
        self.y_test = y_test

    def crear_curva(self) -> pd.DataFrame:
        
        resultados = []
        
        for modelo, pipeline in self.modelos.items():
            check_is_fitted(pipeline)
            probabilidades = pipeline.predict_proba(self.X_test)[:,1]
            precision, recall, thresholds  = precision_recall_curve(self.y_test, probabilidades)
            pr_auc = auc(recall, precision)
            resultados.append({
                'Modelo' : modelo,
                'PR AUC' : pr_auc})
            
            plt.plot(recall,precision, label = f'{modelo} PR AUC = {pr_auc:.4f}')

        plt.xlabel('Recall (sensibilidad)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.grid(True)
        plt.legend()
        plt.show()      

        return(pd.DataFrame(resultados).sort_values("PR AUC", ascending=False).reset_index(drop=True))


class ValidacionCruzadaComparaciones:
    """Ejecuta Validación Cruzada para distintos modelos y compara sus métricas"""
    
    def __init__(self, modelos : dict[str,Pipeline], X_train : pd.DataFrame, y_train : pd.Series, n_splits = 5, random_state = 42) -> None:
        """
        Inicializa la evaluación de los distintos modelos.
        :param modelos: Diccionario que contiene el nombre del modelo y el pipeline del modelo que se desea evaluar.
        :param X_train: Conjunto de entrenamiento de las variables explicativas.
        :param y_train: Variable objetivo correspondiente al conjunto de entrenamiento. 
        :param n_splits: Número de particiones utilizadas en la validación cruzada.
        :param random_state: Semilla utilizada para garantizar la reproducibilidad de las particiones.
        """
        self.modelos = modelos
        self.X_train = X_train
        self.y_train = y_train

        self.tipo_validacion = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        self.scores = None

    def evaluar(self) -> pd.DataFrame:
 
        metricas = {
                "recall": "recall",
                "precision": "precision", 
                "f1": "f1",
                "roc_auc": "roc_auc",
                "average_precision" : "average_precision" }
        
        resultados = []
 
        for modelo, pipeline in self.modelos.items():
            scores = cross_validate(
            estimator=pipeline,
            X=self.X_train,
            y=self.y_train,
            cv=self.tipo_validacion,
            scoring= metricas,
            n_jobs=1,
            return_train_score=False
            )
            
            fila = {'Modelo' : modelo}
        
            for metrica in metricas:
                valores = scores[f'test_{metrica}']
                fila[f"{metrica}_media"] = valores.mean()
                fila[f'{metrica}_std'] = valores.std()
            resultados.append(fila)

        return pd.DataFrame(resultados)

      