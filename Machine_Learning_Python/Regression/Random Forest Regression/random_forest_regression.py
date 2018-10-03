# Random Forest Regression

#importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the dataset into the project
dataset = pd.read_csv('Position_Salaries.csv')

#independent variables
X = dataset.iloc[:, 1:2].values

#dependent variables
y = dataset.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

y_predict = regressor.predict(6.5)

# visualize Random forest regression

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()