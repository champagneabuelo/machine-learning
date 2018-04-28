# Data Preprocessing Template

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Simple Linear Regression to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test results
y_pred = regressor.predict(X_test)

# Visualizing training set results
plt.scatter(X_train, y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs. Experience (Testing Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
