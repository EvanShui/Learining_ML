# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# [:, 0] all rows, first column
# fit the label encoder
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# after this, label encoder has encoded all of the countries and assigned
# France = 0, Germany = 1, Spain = 2
# issue with assigning values to countries, ml models will think that Spain is
# of greater value than Germany or France, because 2 > 1, 2 > 0.
# However, order does not matter. Do implement this, use dummy variables
# Use OneHotEncoder to create dummy variables
# categorical_features: sepcify which column want to 'one hot encode'.
onehotencoder = OneHotEncoder(categorical_features = [0])
# first column - France, second column - Germany, third column - Spain
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
# don't have to use one hot encoder, because yes and no are binary (just 1 and
# 0)
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# X -> independent, y -> dependent
# for the sake of keeping data consistent with udemy course, speicfy
# random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state = 0)
# will output 2 results for test and 8 results for train

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Have to fit the data to the training set and then transform it.
X_train = sc_X.fit_transform(X_train)
# sc_X object is already fitted to the training set, so just need to transform
X_test = sc_X.transform(X_test)
# do we need to fit and transform the dummy variables?          
print(X_train, X_test)
# do we need to feture scale on the dependent variable (y)? No because this is
# a classification problem with a categorical dependent variable 0 - 1 