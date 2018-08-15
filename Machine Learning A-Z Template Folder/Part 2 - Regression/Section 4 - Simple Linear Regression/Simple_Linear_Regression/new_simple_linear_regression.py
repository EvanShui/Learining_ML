# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporitng the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training set and Testing set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
        random_state=0)
# for simple linear regression, scikitlearn already takes care of feature
# scaling

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fits the regressor object to the test data
# model will learn the correlations of the training set to predict the test set
# based off of the independent variable (years of experience)
regressor.fit(X_train, y_train)

# Predicting the Test set results
# vector of predictions for the model
y_pred = regressor.predict(X_test)
# have to compare the values between y_test and y_pred
print(y_pred)
print(y_test)
plt.scatter(X_train, y_train, color='red')
# should display good accuracy
# Visualizing the Training set results
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# now need to display the same linear regression on test set
# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
# Because the regressor was trained with the training data in mind, if we used
# X_test data to plot, it would output the same linear regression, since we
# already trained the regressor with the trained data. All the next line is
# doing is passing in the X_test variables into the regressor, but the
# regressor will still return the same linear regression as with X_train
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
