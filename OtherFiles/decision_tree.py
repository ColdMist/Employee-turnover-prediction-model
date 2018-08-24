# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 03:02:21 2017

@author: Turzo
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('HR_comma_sep.csv')
df.dtypes
df.describe()
all_inputs = df.iloc[:, 1:5].values
all_classes = df.iloc[:,5].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
dtc.score(test_inputs, test_classes)