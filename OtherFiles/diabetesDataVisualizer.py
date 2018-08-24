# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:27:10 2017

@author: Turzo
"""

import pandas as pd
import seaborn as sns

dataset = pd.read_csv('RealdataforDiseas.csv')
dataset.head()
dataset.describe()
sns.pairplot(dataset,hue='Complication')
sns.barplot(x='Complication', y = 'BMI',data = dataset)
sns.barplot(x='Complication', y = 'HbA1c',data = dataset)
sns.boxplot(x='HbA1c', y ='BMI', data= dataset,hue = 'Complication')