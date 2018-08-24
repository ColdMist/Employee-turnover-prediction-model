# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:40:30 2017

@author: Turzo
"""
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import pandas as pd
# Build a classification task using 3 informative features
dataset = pd.read_csv('HR_comma_sep.csv')
dataset['salary'] = dataset['salary'].factorize()[0]
X = dataset.iloc[:, [0, 1,2,3,4, 9]].values
#X = dataset.iloc[:, [3,2]].values
y = dataset.iloc[:, 6].values
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
