import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures    

house_data=pd.read_csv('housing.csv')

X = pd.DataFrame(house_data['RM'])
Y = house_data['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=5)

poly_reg = PolynomialFeatures(degree=2)

X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d,Y_train)

test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)

rms = sqrt(mean_squared_error(Y_test, test_pred))
print(rms)

r2=r2_score(Y_test, test_pred)
print(r2)

plt.scatter(X_test,Y_test)
plt.plot(X_test,test_pred ,color='green')

plt.title("Predicted House Prices")
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.show()

poly_reg = PolynomialFeatures(degree=20)

X_train_20_d = poly_reg.fit_transform(X_train)
X_test_20_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_20_d,Y_train)

test_pred20 = lin_reg.predict(X_test_20_d)
train_pred20 = lin_reg.predict(X_train_20_d)

plt.scatter(X_test,Y_test)
plt.plot(X_test,test_pred20 ,color='green')

plt.title("Predicted House Prices")
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.show()