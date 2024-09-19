# Importamos librerías necesarias (ignorando advertencias de tipo)
import numpy as np  # Librería para trabajar con arrays multidimensionales #type: ignore
import pandas as pd  # Librería para análisis y manipulación de datos#type: ignore
import matplotlib.pyplot as plt  # Librería para creación de gráficos#type: ignore
import seaborn as sns  # Librería para visualización avanzada de datos#type: ignore
from sklearn.model_selection import train_test_split  # Librería para dividir datos en entrenamiento y prueba #type: ignore
from sklearn.preprocessing import StandardScaler  # Librería para escalar características#type: ignore
from sklearn.linear_model import LogisticRegression  # Librería para el modelo de regresión logística#type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  # Librería para métricas de evaluación#type: ignore

# Carga del conjunto de datos
data = pd.read_csv('/usr/local/lib/football_player_stats.csv')

# Exploración inicial de datos
print(data.head())  # Muestra las primeras filas del conjunto de datos
print(data.info())  # Muestra información sobre los tipos de datos y valores nulos
print(data.describe())  # Muestra estadísticas descriptivas de las variables numéricas

# Preprocesamiento de datos
x = data.drop('Goals', axis=1)  # Se separan las características (sin la columna "Goals")
y = data['Goals']  # Se define la variable objetivo ("Goals")

# División de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#  - x_train: Características para entrenamiento
#  - x_test: Características para prueba
#  - y_train: Variable objetivo para entrenamiento
#  - y_test: Variable objetivo para prueba
#  - test_size: Proporción de datos para prueba (20%)
#  - random_state: Semilla aleatoria para garantizar reproducibilidad

# Estandarización de características
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Se ajusta y transforma el conjunto de entrenamiento
x_test_scaled = scaler.transform(x_test)  # Se transforma el conjunto de prueba usando el mismo escalador

# Entrenamiento del modelo de regresión logística
logistic_model = LogisticRegression()
logistic_model.fit(x_train_scaled, y_train)  # Se entrena el modelo con los datos escalados

# Predicción de la variable objetivo
y_pred = logistic_model.predict(x_test_scaled)  # Se predicen los goles usando el modelo entrenado

# Evaluación del modelo
conf_matrix = confusion_matrix(y_test, y_pred)  # Matriz de confusión para ver predicciones correctas e incorrectas

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))  # Se define el tamaño de la figura
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)  # Se crea la matriz de calor
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de confusión')
plt.show()  # Se muestra la matriz de confusión

# Reporte de clasificación con métricas como precisión, recuerdo y F1-score
print(classification_report(y_test, y_pred))

# Cálculo y visualización de la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
