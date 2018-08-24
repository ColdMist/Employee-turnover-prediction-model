# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:32:29 2017

@author: Turzo
"""

# -*- coding: utf-8 -*-
"""
Code to select features and reduce the dimension. The algorithm here
is the Sequential Backward Selection algorithm, following Python
Machine Learning.
Date: Aug 31, 2016
"""
import numpy as np
from sklearn.base import clone
from itertools import combinations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    
    # initializes class - estimator is the ML algorithm of choice, 
    # accuracy_score is the metric to score accuracy by simply comparing the
    # predicted class to the actual class.
    def __init__(self, estimator, k_features, scoring=accuracy_score, 
                 test_size=0.25, random_state=1):
        
        # accuracy scoring function
        self.scoring = scoring     
        
        # clones the estimator for the same parameters without the data
        self.estimator = clone(estimator) 
        
        # number of desired features to reduce the dimensionality to.
        self.k_features = k_features
        
        # spit ratio of test set
        self.test_size = test_size
        
        # seed for random number generator
        self.random_state = random_state            
        
        
    # function to fit the data according to the reduced feature numbers    
    def fit(self, X, y):
        
        # split the data sets into test and train subsets - we need to split
        # the training data rather than our actual test set so that we don't
        # end up using the test data for training.
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size, 
                                 random_state=self.random_state)

        # dimension of training data set - # of features        
        dim = X_train.shape[1]
        
        # range of dim gives the range from 0 to (dim - 1), while tuple
        # generates integers in the range.
        self.indices_ = tuple(range(dim))
        
        # putting the indices in square brackets turns it into a vector.
        self.subsets_ = [self.indices_]
        
        # as defined before, this computes the accuracy of the algorithm for
        # the full feature set.
        score = self._calc_score(X_train, y_train, X_test, y_test, 
                                 self.indices_)                 
        
        # initialize scores_ vector with the initial score.        
        self.scores_ = [score]

        # begin while loop that terminates once the dimension of the feature 
        # space is what we want - k_features.
        while dim > self.k_features:
            
            # intialize scores vectors and indices subsets that is updates as 
            # the feature numbers is reduced.
            scores = []
            subsets = []
            
            # the all (dim - 1) subsets of the feature space of dimension dim
            # are created through combinations from itertools module.
            for p in combinations(self.indices_, r=dim-1):
                
                # compute the prediction accuracy for each index combination.
                score = self._calc_score(X_train, y_train,
                                  X_test, y_test, p)
                                  
                # store the score                                  
                scores.append(score)
                
                # store the index subset
                subsets.append(p)

            # find which position contains the highest score. Note we're not
            # interested in the actual score.
            best = np.argmax(scores)        

            # store the corresponding index as the indices_ variable.                          
            self.indices_ = subsets[best]
            
            # meanwhile, keep track of how the indices evolve with each 
            # decrement of the feature space.
            self.subsets_.append(self.indices_)

            # the dimension has been reduced by 1 via the steps above.                                 
            dim -= 1
            
            # keep track of the scores by adding the score for the best 
            # reduction of the feature space.
            self.scores_.append(scores[best])
            
        # when the while loop is completed, take the last score and store it
        # in the variable k_score_.
        self.k_score_ = self.scores_[-1]
        
        return self
        
    # this function will transform the data set by keeping only the columns
    # with the specified indices. This isn't used anywhere, here but is useful.
    def transform(self, X):
        return X[:, self.indices_]

    # function to find the score of the dataset for specified feature indices.
    # This is used at each step when the feature space has been reduced.
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        
        # fit the data using the learning algorithm of choice.
        self.estimator.fit(X_train[:, indices], y_train) 
        
        # predict the classes.
        y_pred = self.estimator.predict(X_test[:, indices])
        
        # evaluate score
        score = self.scoring(y_test, y_pred)        
        return score
        
        
        
        
        
        