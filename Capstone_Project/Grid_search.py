# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 05:54:55 2017

@author: fanat
"""



# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# TODO: Initialize the classifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_C, X_test, y_test, cv=5)
scores      


# TODO: Create the parameters list you wish to tune
parameters = {'max_depth' : [1,2,3,4,5], 'min_samples_split': [2,3,4,5], 'min_samples_leaf': [2,3,4,5]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(accuracy_score)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))