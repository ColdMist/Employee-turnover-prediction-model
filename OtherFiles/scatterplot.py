# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:05:58 2017

@author: Turzo
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('HR_comma_sep.csv')
dataset['salary'] = dataset['salary'].factorize()[0]
dataset['left'] = dataset['left'].factorize()[0]
X = dataset.iloc[:,6].values
Y = dataset.iloc[:,9].values
Z1 = dataset.iloc[:,0].values
Z2 = dataset.iloc[:,3].values
Z3 = dataset.iloc[:,4].values
Z4 = dataset.iloc[:,5].values
Z5 = dataset.iloc[:,6].values
Z6 = dataset.iloc[:,7].values


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.scatter(X, Y, Z1, c='orange', marker='o')
#ax.scatter(X, Y, Z2, c='green', marker='o')
#ax.scatter(X, Y, Z3, c='blue', marker='o')
#ax.scatter(X, Y, Z4, c='violet', marker='o')

#ax.scatter(X, Y, Z5, c='red', marker='o')
#ax.scatter(X, Y, Z6, c='yellow', marker='o')
ax.set_xlabel('Left')
ax.set_ylabel('Salary')
ax.set_zlabel('Satisfaction')

plt.show()