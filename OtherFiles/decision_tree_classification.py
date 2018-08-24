# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
# Importing the dataset
dataset = pd.read_csv('HR_comma_sep.csv')
dataset['salary'] = dataset['salary'].factorize()[0]
X = dataset.iloc[:, [0, 1,2,3,4, 9]].values
#X = dataset.iloc[:, [3,2]].values
y = dataset.iloc[:, 6].values



'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x1 = LabelEncoder()
X[:,1]=labelEncoder_x1.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])

labelEncoder_x2 = LabelEncoder()
X[:,8]=labelEncoder_x2.fit_transform(X[:,8])
onehotencoder = OneHotEncoder(categorical_features = [8])

X = onehotencoder.fit_transform(X).toarray()
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

'''
labelEncoder_y = LabelEncoder()
labelEncoder_y.fit(X[1])
encoded_Y = labelEncoder_y.transform(X[1])
X[1]=np_utils.to_categorical(encoded_Y)
'''
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
#Previous= X_train.shape

#Select K best features
X_train = SelectKBest(chi2, k=6).fit_transform(X_train, y_train)
X_test = SelectKBest(chi2, k=6).fit_transform(X_test, y_test)

#Now = X_train.shape
#Applying PCA
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''
# Fitting classification methods to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',n_estimators=1000,random_state=1)
classifier.fit(X_train,y_train)

from sklearn import svm
classifier = svm.SVC(kernel='rbf',gamma=0.1)
classifier.fit(X_train,y_train)
 
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(X_train,y_train)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
#n_error_train = y_pred_train[y_pred_train == -1].size

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print ('RF accuracy: TRAINING', classifier.score(X_train,y_train))
print ('RF accuracy: TESTING', classifier.score(X_test,y_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
c = np.where(y_pred>0)

plt.scatter(X_test[:, 0], X_test[:, 1], s=40, c='gray')
plt.scatter(X_test[c, 0], X_test[c, 1], s=160, edgecolors='red',
           facecolors='none', linewidths=2, label='Class 1')
'''



'''
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
'''