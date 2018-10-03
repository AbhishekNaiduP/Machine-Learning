# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset.head()

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# predicting a new result

y_Pred = regressor.predict(6.5)

#Visualizing the Decision Tree model

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()

from sklearn.tree import DecisionTreeClassifier 

clf_entropy = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_leaf = 3, random_state = 0)
clf_entropy.fit(X, y)

y_predict = clf_entropy.predict(6.5)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, clf_entropy.predict(X_grid), color = 'blue')
plt.show()

predict_y = clf_entropy.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = (accuracy_score(y_test, predict_y))*100

