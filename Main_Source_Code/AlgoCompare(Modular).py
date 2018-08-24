# Feature Analysis of employee turnover using Machine Learning Approach
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from SBSAlgorith_working import SBS
from sklearn.neighbors import KNeighborsClassifier
# Importing the dataset
# Change to preprocessed dataset
dataset = pd.read_csv('HR_comma_sep(SalaryManipulated).csv')
#preprocessing the data 
#dataset['salary'] = dataset['salary'].factorize()[0]
dataset['type'] = dataset['type'].factorize()[0]    
dataset['satisfaction_level']=dataset['satisfaction_level']*100
dataset['last_evaluation']=dataset['last_evaluation']*100
#newDataset = dataset[['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','type','salary']].copy()
newDataset = dataset[['satisfaction_level','last_evaluation', 'average_montly_hours','time_spend_company','salary']].copy()
#functions############################################################
def SBSShow(X_train,y_train):
    #KNN = KNeighborsClassifier(n_neighbors=3)
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
    sbs = SBS(classifier, k_features=1)
    sbs.fit(X_train, y_train)
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker = 'o')
    plt.ylim([0.7,1.1])
    plt.ylabel('accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()
    #print(sbs.k_feature_idx_)
    #k=list(sbs.subsets_)
    #print(newDataset.columns[0:][k])
    print('Indices: ', sbs.subsets_)
    print('Scores: ', sbs.scores_)
    #print(newDataset.columns[1:][k])
    
def RandomForestImportanceShow():
    feat_labels = newDataset.columns[0:]
    forest = RandomForestClassifier(criterion='entropy',n_estimators=1000, random_state=1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(0,5):
        print("%2d) %-*s %f" % (f+1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

def doChi2Calculation(X_train,y_train,X_test,y_test,k_num):
    selector = SelectKBest(chi2, k=k_num).fit(X_train,y_train)
    X_train = selector.transform(X_train) # not needed to get the score
    X_test = selector.transform(X_test)
    scores = selector.scores_
    return X_train,X_test,scores
#functions###############################################################

# Importing the dataset
#dataset = pd.read_csv('HR_comma_sep(SalaryManipulated).csv')
#preprocessing the data 
#dataset['salary'] = dataset['salary'].factorize()[0]    
#dataset['satisfaction_level']=dataset['satisfaction_level']*100
#dataset['last_evaluation']=dataset['last_evaluation']*100
#newDataset = dataset[['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours','time_spend_company','salary']].copy()
#dataset['salary'] = dataset['salary'].factorize()[0]    
#X = dataset.iloc[:, [0, 1,2,3,4,5,7,8, 9]].values
X = dataset.iloc[:, [0,1,3,4,9]].values
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

from sklearn.preprocessing import RobustScaler
rb = RobustScaler()
X_train=rb.fit_transform(X_train)
X_test = rb.transform(X_test)
'''

#from sklearn.preprocessing import MinMaxScaler
#MMS = MinMaxScaler()
#X_train= MMS.fit_transform(X_train)
#X_test = MMS.transform(X_test)
#Select K best features

'''
X_train = SelectKBest(chi2, k=6).fit_transform(X_train, y_train)
X_test = SelectKBest(chi2, k=6).fit_transform(X_test, y_test)
'''
#Visualize SBS plot
SBSShow(X_train,y_train)
#chi2 selector train test formulation
X_train,X_test,scores=doChi2Calculation(X_train,y_train,X_test,y_test,3)
#Random Forest importance show do not use this if alrady used doChi2Calculation
RandomForestImportanceShow()
#Get the scores of the best features
# write k= 'all' to get the scores of all the features
#selector = SelectKBest(chi2, k=3).fit(X_train,y_train)
#X_train = selector.transform(X_train) # not needed to get the score
#X_test = selector.transform(X_test)
#scores = selector.scores_
#Univariate Feature Selcection
#from sklearn.feature_selection import SelectPercentile, f_classif
#selector = SelectPercentile(f_classif, percentile=50)
#selector.fit(X_train, y_train)
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
#call SBS with Dtree to show SBS result
#SBSShow()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000,random_state=0)
classifier.fit(X_train,y_train)
#Show Random Forest Importance
#RandomForestImportanceShow()

#print(X_train.Shape)
from sklearn import svm
classifier = svm.SVC(kernel='rbf',gamma=0.1)
classifier.fit(X_train,y_train)
 
classifier = MLPClassifier()
classifier.fit(X_train,y_train)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Get the accuracy 
print ('accuracy: TRAINING', classifier.score(X_train,y_train))
print ('accuracy: TESTING', classifier.score(X_test,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plot caliberation curve (is our future work)
def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    #isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    #sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
   # lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name)]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

# Plot calibration curve for Gaussian Naive Bayes
#plot_calibration_curve(DecisionTreeClassifier(criterion = 'entropy', random_state = 1),"Decision Tree",1)
#plot_calibration_curve(RandomForestClassifier(criterion='entropy',n_estimators=1000,random_state=1),"Random Forest",2)
#plot_calibration_curve(svm.SVC(kernel='rbf',gamma=0.1),"SVM",3)
plot_calibration_curve(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2)),"MLPClassifier",4)
#plot_calibration_curve(GaussianNB(),"GaussianNB",5)
# Plot calibration curve for Linear SVC
#plot_calibration_curve(LinearSVC(), "SVC", 2)
plt.show()

