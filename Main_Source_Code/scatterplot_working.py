# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:05:58 2017

@author: Turzo
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv('GaussianNB.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values
Z1 = dataset.iloc[:,2].values
'''
Z2 = dataset.iloc[:,3].values
Z3 = dataset.iloc[:,4].values
Z4 = dataset.iloc[:,5].values
Z5 = dataset.iloc[:,6].values
Z6 = dataset.iloc[:,7].values
'''
leftCounter = np.where(dataset['Prediction']>0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

'''
ax.scatter(X[leftCounter,0], Y[leftCounter,1],Z1, s=160, edgecolors='red',
           facecolors='none', linewidths=2, label='Class 1')
'''
count= np.where(dataset['Prediction']>0,'r','b')
ax.scatter(X, Y, Z1, c=count, marker='o')



#ax.scatter(X, Y, Z3, c='blue', marker='o')
#ax.scatter(X, Y, Z4, c='violet', marker='o')

#ax.scatter(X, Y, Z5, c='red', marker='o')
#ax.scatter(X, Y, Z6, c='yellow', marker='o')
ax.set_xlabel('Satisfaction_Level')
ax.set_ylabel('Work Hour Per month')
ax.set_zlabel('Time Spent')

plt.show()