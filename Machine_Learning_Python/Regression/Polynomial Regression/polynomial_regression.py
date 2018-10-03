# Polynomial Regression

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures

#transform X into Polynomial Matrix
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)

# this regression is trained by X_poly that means by polynomial matrix
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#visualizing the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X))
plt.xlabel = 'Experience Level'
plt.ylabel = 'Salaries'
plt.title = 'Salaries vs Experience'
plt.show()

#Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)))
plt.xlabel = 'Experience Level'
plt.ylabel = 'Salaries'
plt.title = 'Salaries vs Experience'
plt.show()

#predicting a new results with Linear Regression
lin_reg.predict(6.5)

#predicting a new results with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))