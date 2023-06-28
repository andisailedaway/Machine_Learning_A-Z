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

print(ind)