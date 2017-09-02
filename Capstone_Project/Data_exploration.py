# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:49:17 2017

@author: fanat
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
import visuals as vs

# Pretty display for notebooks

# Load the cancer dataset
data = pd.read_csv("data.csv")

# Success - Display the first record
display(data.head(n=1))


# TODO: Total number of records
n_records = len(data)

# TODO: Number of records where the diagnosis is malignant
n_malignant = len(data['diagnosis'][data['diagnosis']=='M'])

# TODO: Number of records where the diagnosis is benign
n_benign = len(data['diagnosis'][data['diagnosis']=='B'])

# TODO: Percentage of records where the diagnosis is malignant
malignant_percentage = (float(n_malignant)/float(n_records))*100

# Print the results
print "Total number of records: {}".format(n_records)
print "Number of records where the diagnosis is malignant: {}".format(n_malignant)
print "Number of records where the diagnosis is benign: {}".format(n_benign)
print "Percentage of records where the diagnosis is malignant: {:.2f}%".format(malignant_percentage)

#%%

# Split the data into features and target label
diagnosis_label = data['diagnosis']
features_raw = data.drop(['diagnosis', 'Unnamed: 32'], axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# Log-transform the skewed features
skewed = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)

#%%
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))


#%%

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
diagnosis = diagnosis_label  
diagnosis =diagnosis.replace(to_replace = 'B', value = 0)
diagnosis =diagnosis.replace(to_replace = 'M', value = 1)


# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
print encoded


#%%

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, diagnosis, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
print "Testing set has {} samples.".format(y_train.shape[0])
print "Testing set has {} samples.".format(y_test.shape[0])


#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
# TODO: Calculate accuracy
accuracy = malignant_percentage/100

precision = accuracy
recall = 1

# TODO: Calculate F-score using the formula above for beta = 0.5
fscore = (1+np.square(0.5))*((precision*recall)/((np.square(0.5)*precision)+recall))

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)


#%%

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results


#%%

# TODO: Import the three supervised learning models from sklearn
from sklearn import tree

# TODO: Initialize the three models
clf_C = tree.DecisionTreeClassifier(random_state=0)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = X_train.shape[0]/100
samples_10 = X_train.shape[0]/10
samples_100 = X_train.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


#%%


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_C, X_test, y_test, cv=5)
scores      


#%%
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,clf.predict(X_test))

FN = matrix[1][0]
print(FN)




