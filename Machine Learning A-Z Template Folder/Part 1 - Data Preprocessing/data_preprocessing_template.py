# -------------------------------------------------
#           Impoting the Libraries
# -------------------------------------------------

# Data Preprocessing Template

# Importing the libraries
# mathematical library
import numpy as np

# plotting library
import matplotlib.pyplot as plt

# dataframe library for dataset management
import pandas as pd


# -------------------------------------------------
#           Importing the Dataset
# -------------------------------------------------

# ----------------------------------
#           Main Code
# ----------------------------------

# Importing the dataset
dataset = pd.read_csv('Data.csv')

print(dataset)

# iloc gets rows or columns at particular positions in index
# Parameters: 
# iloc[lst_row, lst_col]
# lst_row: list selection notation for selecting 0 through x-1 rows
# lst_col: list selection notation for selecting 0 through x-1 columns
# .values: creates values 
# X is type numpy.ndarray (similar to list but can do manipulations with numpy
# no commas to separate between entries in list, only spaces
X = dataset.iloc[:, :-1].values

# creates an array of values of the last column (last column's index is 3) 
# prepared the data by this point
y = dataset.iloc[:, 3].values
print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# ----------------------------------
#           Test space
# ----------------------------------


# will return a dataframe, by appending values to the end of the statement will
# then return an array of values
fun = dataset.iloc[1:, 1:-1]

print(fun)
