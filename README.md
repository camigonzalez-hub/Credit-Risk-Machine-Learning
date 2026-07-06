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