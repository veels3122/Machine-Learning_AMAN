import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt # type: ignore

House_Area = np.array([60,75,90,55,100,85,120,80,70,95,110,65,105,130,140,50,160,125,140,135,45,150,125,115,80,90,100])
Sale_Price = np.array([200,250,275,180,320,270,350,260,240,290,330,220,310,370,400,170,420,360,380,370,160,390,340,320,250,275,310])

x = House_Area.reshape(-1,1)
y = Sale_Price

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)  

r2 = model.score(x_test, y_test)
coefficients = model.coef_[0]
intercept = model.intercept_

print("Linear Regression")
print("R2 Score:", r2)
print("Coefficients:", coefficients)
print("Intercept:", intercept)

print("predicted Vs Current Price")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual Price: {actual:.2f}, Predicted Price: {predicted:.2f}")

print(" ")

plt.scatter(x,y, color='blue', label='Actual Price')
plt.plot(x_test, y_pred, color='red', label='Predicted Price')
plt.xlabel('House Area')
plt.ylabel('Sale Price')
plt.title('House Area vs Sale Price')
plt.legend()
plt.show()