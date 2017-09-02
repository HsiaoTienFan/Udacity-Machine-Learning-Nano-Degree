# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:49:17 2017

@author: Hsiao-Tien Fan


"""
#%% Import
import numpy as np
import pandas as pd
import visuals as vs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#%% Read in data
data = pd.read_csv("data.csv")
n_records = len(data)
n_malignant = len(data['diagnosis'][data['diagnosis']=='M'])
n_benign = len(data['diagnosis'][data['diagnosis']=='B'])
malignant_percentage = (float(n_malignant)/float(n_records))*100

# Print the results
print "\nTotal number of records: {}".format(n_records)
print "Number of records where the diagnosis is malignant: {}".format(n_malignant)
print "Number of records where the diagnosis is benign: {}".format(n_benign)
print "Percentage of records where the diagnosis is malignant: {:.2f}%".format(malignant_percentage)

#%% Preprocess data

# Split the data into features and target label
diagnosis_label = data['diagnosis']
# Remove non-features
features = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1)
# Log-transform the features
skewed = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
features[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Normalize features to 0-1
scaler = MinMaxScaler()
numerical = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
features[numerical] = scaler.fit_transform(data[numerical])

# Encode the labels
diagnosis = diagnosis_label  
diagnosis =diagnosis.replace(to_replace = 'B', value = 0)
diagnosis =diagnosis.replace(to_replace = 'M', value = 1)

#%% Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, diagnosis, test_size = 0.2, random_state = 0)

# Display the results of the split
print "\nTraining set input has {} samples.".format(X_train.shape[0])
print "Training set output has {} samples.".format(y_train.shape[0])
print "Testing set input has {} samples.".format(X_test.shape[0])
print "Testing set output has {} samples.".format(y_test.shape[0])


#%% Calculate metrics
accuracy = malignant_percentage/100

precision = accuracy
recall = 1
F1 = 2*(precision*recall)/(precision+recall)


# Display the results 
print "\nNaive Predictor Accuracy: [Accuracy score: {:.4f}]".format(accuracy)
print "Naive Predictor F1: [F1 score: {:.4f}]".format(F1)



#%% Setup and train decision tree
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print "\nAccuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F1 score on testing data: {:.4f}".format(f1_score(y_test, predictions))

#%% Calculate and display the cross validation score
scores = cross_val_score(clf, X_test, y_test, cv=5)
print "\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
scores2 = cross_val_score(clf, X_test, y_test, cv=5, scoring='f1_macro')
print "F1: %0.2f (+/- %0.2f)" % (scores2.mean(), scores.std() * 2)

#%% Use grid search to optimize the decision tree

parameters = {'max_depth' : [1,2,3,4,5], 'min_samples_split': [2,3,4,5], 'min_samples_leaf': [2,3,4,5]}
scorer = make_scorer(f1_score)
grid_search = GridSearchCV(clf, parameters, scorer)
grid_fit = grid_search.fit(X_train, y_train)
optimal_clf = grid_search.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
optimal_predictions = optimal_clf.predict(X_test)

# Report the before-and-afterscores
print "\nUnoptimized model"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F1 score on testing data: {:.4f}".format(f1_score(y_test, predictions))
print "\nOptimized Model"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, optimal_predictions))
print "F1 score on testing data: {:.4f}".format(f1_score(y_test, optimal_predictions))
print "\nThe optimized configuration of the decision tree:"
print optimal_clf



#%% Finding the top important features in the model
top_features = clf.feature_importances_
vs.feature_plot(top_features, X_train, y_train)


#%% Train a new model only with top five features
X_train_reduced = X_train[X_train.columns.values[(np.argsort(top_features)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(top_features)[::-1])[:5]]]

# Reuse previous optimal parameters and train with top features
clf = (clone(optimal_clf)).fit(X_train_reduced, y_train)

# New prediction
new_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Model trained on full data"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, optimal_predictions))
print "F1 score on testing data: {:.4f}".format(f1_score(y_test, optimal_predictions))
print "\nModel trained on reduced data"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, new_predictions))
print "F1 score on testing data: {:.4f}".format(f1_score(y_test, new_predictions))


#%% Calculate the confusion matrix and the false negative and false positive

tn, fp, fn, tp = confusion_matrix(y_test,clf.predict(X_test_reduced)).ravel()

print "\nFalse Negative: %0.2f" %(fn*100/float(len(X_test_reduced)))
print "False Positive: %0.2f" %(fp*100/float(len(X_test_reduced)))
