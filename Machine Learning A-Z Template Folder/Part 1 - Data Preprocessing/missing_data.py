# -------------------------------------------------
#           Missing Data
# -------------------------------------------------

# Methods of handling missing data 
# Removing entries with missing data
# Replace the missing data with the mean of the entire column

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print(X)

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Parameters: 
# missing_values: placeholder value for a missing value
# strategy: imputation strategy
# axis: if axis = 0, impute along columns, 
#       if axis = 1, impute along rows
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# will fit the imputer on the columns where there is missing data (only columns
# 1 to 3)
# X[:, 1:3] -> all rows, columns 1 and 2
# fit imputer object to the matrix 'X'
imputer = imputer.fit(X[:, 1:3])
print(imputer)

# replace missing data with mean (transform -> replaces missing data with mean
# of column
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)