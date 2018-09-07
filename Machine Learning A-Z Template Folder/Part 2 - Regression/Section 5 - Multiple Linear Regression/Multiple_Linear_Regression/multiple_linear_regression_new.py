# Multiple Linear Regression 

# Importing the Libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# just fits the encoder onto the data
# categorical variable is located in column 3
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
print(X)
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
print(X)

# Avoiding the Dummy Variable Trap
# Just removed the first column from X, want to take all columns from 1 to the end of the list.
X = X[:, 1:]

#splitting the dataset into the Training set an test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#have to encode the categorical variables (state variable)
#use label encoder / one hotencoder to create dummy variables

