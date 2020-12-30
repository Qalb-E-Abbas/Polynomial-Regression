# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values



# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


pol_reg.predict(poly_reg.fit_transform([[5.5]]))

