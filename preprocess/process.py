#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("preprocessingdata.csv")
#print(dataset)

#select independent variables
ind = dataset.iloc[:, :-1].values
print(ind)

# ind2 = dataset[["Country", "Age", "Salary"]]
# print(ind2)

#select dependent variables
dep = dataset.iloc[:, -1].values
print(dep)

# dep2 = dataset[["Purchased"]]
# print(dep2)

#replace NaN with average column values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(ind[:,1:3])
ind[:,1:3] = imputer.transform(ind[:,1:3])

#encoding independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#reaminder = passthrough says that I want to keep the other columns not being encoded
ind = np.array(ct.fit_transform(ind))
#force output to be Numpy array
print(ind)

#encode dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dep = le.fit_transform(dep)
print(dep)

#splitting data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ind, dep, test_size=0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#feature scaling
#apply after splitting data into training and test b/c you want your scaling to only apply to a single dataset
from sklearn.preprocessing import  StandardScaler
sc = StandardScaler()
x_train[: , 3:] = sc.fit_transform(x_train[: , 3:])
x_test[: , 3:] = sc.transform(x_test[: , 3:])

print(x_train)
print(x_test)