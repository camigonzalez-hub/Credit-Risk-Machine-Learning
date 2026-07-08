### Predicción de Riesgo Crediticio utilizando Regresión Logística.

Este proyecto desarrolla un modelo de regresión logística para predecir si un solicitante de un crédito representa un riesgo o no. Se buscará clasificar a cada solicitud de crédito como Riesgosa y No Riesgosa. El proyecto se desarrollará utilizando el dataset German Credit (credit-g). Lo pueden encontrar en: https://www.openml.org/search?type=data&status=active&id=31 

## Estructura del proyecto:
- Realizar un análisis exploratorio de los datos (EDA)
- Prepocesar los datos (realizar las transformaciones necesarias)
- Contrucción de un Pipeline, que permita la ejecución del modelo en los conjuntos de entrenamiento y de test.
- Evaluación del modelo mediante métricas, y optimización del mismo.
- Re-evaluación final.

# Información sobre el Dataset.
German Credit dataset: Clasifica personas descritas por un set de atributos como buen o mal riesgo crediticio. 

Es 5 veces peor clasificar a un usuario como bueno cuando es malo, que clasificarlo como malo cuando es bueno.

Descripcion de variables: 
1.Status of existing checking account, in Deutsche Mark.
2.Duration in months
3.Credit history (credits taken, paid back duly, delays, critical accounts)
4.Purpose of the credit (car, television,...)
5.Credit amount
6.Status of savings account/bonds, in Deutsche Mark.
7.Present employment, in number of years.
8.Installment rate in percentage of disposable income
9.Personal status (married, single,...) and sex
10.Other debtors / guarantors
11.Present residence since X years
12.Property (e.g. real estate)
13.Age in years
14.Other installment plans (banks, stores)
15.Housing (rent, own,...)
16.Number of existing credits at this bank
17.Job
18.Number of people being liable to provide maintenance for
19.Telephone (yes,no)
20.Foreign worker (yes,no)

Número de instancias: 1000
Número de columnas: 21

### Decisiones respecto al modelo, el preprocesamiento de los datos y su regularización.

**Definición de la clase positiva:** La regresión logística modela la probabilidad de pertenecer a la clase positiva. Por este motivo, la elección de dicha clase determina la interpretación de los coeficientes estimados por el modelo. Dado que el objetivo es identificar clientes con mayor riesgo de generar un mal crédito, se definió la categoría "bad" como la clase positiva, asignándole el valor 1, mientras que "good" se asignó al valor 0. Esta decisión permite una interpretación más intuitiva de los resultados: un coeficiente positivo indica un aumento en la probabilidad de que el cliente pertenezca a la categoría "bad" (mayor riesgo de incumplimiento), mientras que un coeficiente negativo indica una disminución de dicha probabilidad.

**Manejo de las variables categóricas:** Las variables categóricas fueron codificadas mediante One-Hot Encoding, definiendo explícitamente una categoría de referencia para evitar multicolinealidad perfecta y facilitar la interpretación de los coeficientes de la regresión logística. Se eligió como categoría de referencia la más frecuente por ser la mejor representada en la muestra, lo que facilita la interpretación de los coeficientes y permite que todas las comparaciones se realicen respecto de la categoría predominante.