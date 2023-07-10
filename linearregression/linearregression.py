import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("linearregression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#visualize training set results
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs. Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualize test set results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_test, regressor.predict(x_test), color = "blue")
plt.title("Salary vs. Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

#Finding final linear regression coefficients
print(regressor.coef_)
print(regressor.intercept_)
