import numpy as np  # Biblioteca NumPy para operaciones numéricas # type: ignore
import matplotlib.pyplot as plt  # Biblioteca Matplotlib para graficar # type: ignore
from sklearn.model_selection   # type: ignore
 import train_test_split  # Para dividir los datos en entrenamiento y prueba # type: ignore
from sklearn.linear_model import LinearRegression  # Para crear el modelo de regresión lineal # type: ignore

# Datos de ejemplo: Área de la casa y precio de venta
House_Area = np.array([60, 75, 90, 55, 100, 85, 120, 80, 70, 95, 110, 65, 105, 130, 140, 50, 160, 125, 140, 135, 45, 150, 125, 115, 80, 90, 100])
Sale_Price = np.array([200, 250, 275, 180, 320, 270, 350, 260, 240, 290, 330, 220, 310, 370, 400, 170, 420, 360, 380, 370, 160, 390, 340, 320, 250, 275, 310])

# Convertir los datos a una matriz para que sean compatibles con el modelo
x = House_Area.reshape(-1, 1)  # -1 indica que se calcule automáticamente el número de filas
y = Sale_Price

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo   con los datos de entrenamiento
model.fit(x_train, y_train)

# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(x_test)  
r2 = model.score(x_test, y_test)# Evaluar el modelo calculando el coeficiente de determinación R²


# Obtener los coeficientes y la intersección del modelo
coefficients = model.coef_[0]
intercept = model.intercept_

# Imprimir los resultados
print("Regresión Lineal")
print("R² Score:", r2)
print("Coeficientes:", coefficients)
print("Intersección:", intercept)

# Comparar los precios reales y predichos
print("Precios Reales vs. Precios Predichos")
for actual, predicted in zip(y_test, y_pred):
    print(f"Precio Real: {actual:.2f}, Precio Predicho: {predicted:.2f}")

# Graficar los datos y la línea de regresión
plt.scatter(x, y, color='blue', label='Precio Real')
plt.plot(x_test, y_pred, color='red', label='Precio Predicho')
plt.xlabel('Área de la Casa')
plt.ylabel('Precio de Venta')
plt.title('Área de la Casa vs. Precio de Venta')
plt.legend()
plt.show()