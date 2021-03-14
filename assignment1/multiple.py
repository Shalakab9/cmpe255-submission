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

X = house_data.iloc[:, :-1]
Y = house_data.iloc[:, -1]

feature_set = house_data[['RM', 'PTRATIO', 'LSTAT']]

X_train, X_test, Y_train, Y_test = train_test_split(house_data[['RM', 'PTRATIO', 'LSTAT']], house_data['MEDV'])
multiple_reg = LinearRegression()
multiple_reg.fit(X_train, Y_train)
y_pred = multiple_reg.predict(X_test)

plt.scatter(X_test['RM'], Y_test)
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.plot(X_test['RM'], y_pred, color="green")
plt.show()

rms = sqrt(mean_squared_error(Y_test, y_pred))
print(rms)

r2=r2_score(Y_test, y_pred)
print(r2)

adjusted_r2=1 - (1-r2)*(len(Y_test)-1)/(len(Y_test)-X_train.shape[1]-1)
print(adjusted_r2)

