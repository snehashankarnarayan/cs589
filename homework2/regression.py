#!/usr/bin/python

import numpy as np
import sys
import math
from time import time
import csv
from pprint import pprint
from sklearn import neighbors
from sklearn import svm
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.linear_model import *
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import *
from sklearn.cross_validation import *
import itertools

# Dataset locations
CRIME_TEST = ""
CRIME_TRAIN = ""
FOREST_TEST = ""
FOREST_TRAIN = ""

dataset = ""


#Function that stores the paths of the dataset in my machine
def load_mach():
    global CRIME_TEST, CRIME_TRAIN, FOREST_TEST, FOREST_TRAIN
    CRIME_TEST = '/Users/snehas/vagroot/shared_files/data/HW2DataDistribute/CommunityCrime/test_distribute.npy'
    CRIME_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW2DataDistribute/CommunityCrime/train.npy'

    FOREST_TEST = '/Users/snehas/vagroot/shared_files/data/HW2DataDistribute/ForestFires/test_distribute.npy'
    FOREST_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW2DataDistribute/ForestFires/train.npy'

#Function that stores the paths of the dataset in my VM
def load_vm():
    global CRIME_TEST, CRIME_TRAIN, FOREST_TEST, FOREST_TRAIN
    CRIME_TEST = '/home/vagrant/Desktop/shared_files/data/HW2DataDistribute/CommunityCrime/test_distribute.npy'
    CRIME_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW2DataDistribute/CommunityCrime/train.npy'

    FOREST_TEST = '/home/vagrant/Desktop/shared_files/data/HW2DataDistribute/ForestFires/test_distribute.npy'
    FOREST_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW2DataDistribute/ForestFires/train.npy'

#Function that stores the paths of the dataset in the evaluators VM
def load_eval():
    global CRIME_TEST, CRIME_TRAIN, FOREST_TEST, FOREST_TRAIN
    CRIME_TEST = '/vagrant/shared_files/data/HW2DataDistribute/CommunityCrime/test_distribute.npy'
    CRIME_TRAIN = '/vagrant/shared_files/data/HW2DataDistribute/CommunityCrime/train.npy'

    FOREST_TEST = '/vagrant/shared_files/data/HW2DataDistribute/ForestFires/test_distribute.npy'
    FOREST_TRAIN = '/vagrant/shared_files/data/HW2DataDistribute/ForestFires/train.npy'


#Function to spit the output in an appropriately named CSV file
def output(data, regression_method):
    with open("output/" + dataset + "_" + regression_method + ".csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'Target'])
        for i in range(0, len(data)):
            writer.writerow(["{0:0.1f}".format(float(i+1)), "{0:0.1f}".format(data[i])])

#Get the hyperparameters of the Support vector regression
def get_hyperparams_svr(x_data, y_class):
    scores = dict()

    #Specify the values being searched over
    kernel_list = ['rbf']
    gamma_list = [0.0001, 0.001, 0.001, 0, 0.01]
    epsilon_list = [0, 0.01, 0.1, 0.2, 0.5]
    C_list = [1,0.2,0.3,0.4,0.5]

    #Variables to store the final hyperparameters
    min_score = 9999.00
    max_kernel = ""
    max_gamma = -1.0
    max_epsilon = -1.0
    max_C = -1.0

    #For every set of hyperparameters, perform cross-validation using MAE as scoring function
    for kernel in kernel_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                for C in C_list:
                    clf = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)
                    print "Doing " + kernel + ", " + str(gamma) + ", " + str(C) + ", " + str(epsilon)
                    score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_absolute_error').mean()
                    scores[kernel,gamma,epsilon,C] = score
                    #Get least error
                    if(score < min_score):
                        min_score = score
                        max_kernel = kernel
                        max_epsilon = epsilon
                        max_C = C
                        max_gamma = gamma


    #Printing the output
    print max_C, max_gamma, max_kernel, max_epsilon
    return max_C, max_gamma, max_kernel, max_epsilon

#Get hyperparameters for KNN Regression
def get_hyperparams_knn(x_data, y_class):
    scores = dict()

    #Hyperparameter values searched over
    k_list = range(1,10)
    distance_list = ['manhattan', 'euclidean', 'chebyshev']
    weight_list = ['uniform', 'distance']

    #Variables to store the final hyperparameters
    min_score = 9999.00
    max_k = 0
    max_distance = ""
    max_weight = ""

    #For every set of hyperparameters, perform cross-validation using MSE as scoring function
    for k in k_list:
        for distance in distance_list:
            for weight in weight_list:
                clf = neighbors.KNeighborsRegressor(n_neighbors = k, weights = weight, metric = distance)
                score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_squared_error').mean()
                scores[k, distance, weight] = score
                #Get least error
                if(score < min_score):
                    min_score = score
                    max_k = k
                    max_distance = distance
                    max_weight = weight

    #Printing and returing output
    print max_k, max_distance, max_weight
    return max_k, max_distance, max_weight

#Get hyperparameters for Random Forest Regression
def get_hyperparams_rf(x_data, y_class):
    scores = dict()

    #Hyperparameter values searched over
    depth_list = [None, 10, 50, 100, 250, 500, 1000]
    n_estimators = [10,20]
    features = ['auto', 'sqrt', 'log2']

    #Variables to store the final hyperparameters
    min_score = 9999.00
    max_n = 0
    max_depth = 0
    max_feature = ""

    #For every set of hyperparameters, perform cross-validation using MSE as scoring function
    for d in depth_list:
        for n in n_estimators:
            for f in features:
                clf = ensemble.RandomForestRegressor(n_estimators=n, max_depth=d, max_features=f)
                score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_squared_error').mean()
                #Get least error
                if(score < min_score):
                    min_score = score
                    max_n = n
                    max_depth = d
                    max_feature = f

    #Printing and returing output
    print max_n, max_depth, max_feature
    return max_n, max_depth, max_feature

#Functions for output transformation
#Doing y = e^y - 0.01
def anti_transform(data):
    for i in range(0, len(data)):
        data[i] = math.exp(data[i]) - 0.01
    return data

#Doing y = log(e + 0.01)
def transform(data):
    for i in range(0, len(data)):
        data[i] = math.log(data[i] + 0.01)
    return data

#Backward stepwise selection
def backwardStepwiseSelection(regress, data, y_class):
    print "Backward stepwise selection"
    min_score = 99999
    max_k = []
    scores = dict()

    #Run for k = p-1 to 1; p = number of predictors
    for k in range(len(data[0]) -1, 1, -1):
        min_step_score = 99999
        step_data = np.delete(data, k, 1)
        score = -cross_val_score(regress, step_data, y_class, scoring='mean_squared_error').mean()
        if(score < min_step_score):
            min_step_score = score
            max_k = [k]
        #Run from 0 to k, delete each feature and see which has the least MAE
        for i in range(0, k):
            print "Doing: " + str(k)
            x_data = np.delete(step_data, i, 1)
            score = -cross_val_score(regress, x_data, y_class, scoring='mean_squared_error').mean()
            #Get least error
            if(score < min_step_score):
                min_step_score = score
                max_k = [k,i]
        scores[k] = min_step_score
        if(min_step_score < min_score):
            min_score = min_step_score

    #Return features to be deleted
    return max_k

#Best subset selection
def bestSubsetSelection(regress, data, y_class):
    min_score = 99999
    num_predictors = len(data[0]);
    subset = []
    scores = dict()
    for k in range(0, num_predictors):
        print "Doing: " + str(k)
        #Get all combinations of pCk
        arr = itertools.combinations(range(num_predictors), k)
        combinations = list(arr)
        min_step_score = 99999
        #Run through all combinations and choose the one with least MAE
        for item in combinations:
            print item
            x_data = np.delete(data, item, 1)
            score = -cross_val_score(regress, x_data, y_class, cv = 5, scoring='mean_absolute_error').mean()
            scores[k] = score
            #Get least error
            if(score < min_step_score):
                min_step_score = score
                subset = item
        scores[k] = min_step_score
        if(min_step_score < min_score):
            min_score = min_step_score

    #Printing the results
    for k in range(0, num_predictors):
        print str(k+1) + " " + str(scores[k])
    print min_score, subset

    #Return features to be deleted
    return subset


#Regression function. This is the main regression function, accepts the method,
## train data location and test data location
def regression(test_location, train_location, regression_method):
    train = np.load(train_location)

    #Get the required training data
    x_data = train[:, 1:]  #All but the first column
    y_class = train[:, 0]

    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    #Pipeline crime:
    if(regression_method == "crime_pipe"):
        regress = ensemble.BaggingRegressor(base_estimator = ensemble.RandomForestRegressor(random_state=1000, max_features='log2', max_depth=250), n_estimators = 20, random_state=1000)

    #Pipeline forest
    if(regression_method == "forest_pipe"):
        C, gamma, kernel, epsilon = get_hyperparams_svr(x_data, y_class)
        regress = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)

    #SVR
    if(regression_method == "svr"):
        regress = SVR()

    #KNN
    elif( regression_method == "knn"):
        regress = neighbors.KNeighborsRegressor()

    #Lasso linear regression
    elif (regression_method == "lin"):
        regress = LassoLarsCV()

    #Decision tree regression
    elif (regression_method == "tree"):
        regress = DecisionTreeRegressor()

    #Place holder for any random method I want to try from sklearn.ensemble
    elif (regression_method == "rand"):
        regress = ensemble.RandomForestRegressor()
        max_k = backwardStepwiseSelection(regress, x_data, y_class)
        x_data = np.delete(x_data, max_k, 1)
        test_data = np.delete(test_data, max_k, 1)
        max_n, max_depth, max_feature = get_hyperparams_rf(x_data, y_class)
        regress = ensemble.RandomForestRegressor(n_estimators=max_n, max_depth=max_depth, max_features=max_feature)

    #Do feature selection
    #max_k = backwardStepwiseSelection(regress, x_data, y_class)
    #x_data = np.delete(x_data, max_k, 1)
    #test_data = np.delete(test_data, max_k, 1)

    print 'Done loading data'
    #Train the regression model and compute time taken
    t1 = time()
    regress.fit(x_data, y_class)
    t2 = time()

    print 'Training time taken in seconds: %f' % (t2 - t1)

    #Do the predictions and compute time taken
    t3 = time()
    out = regress.predict(test_data)
    t4 = time()

    print 'Prediction time taken in seconds: %f' % (t4 - t3)

    #Spit the output into CSV file
    output(out, regression_method)

def run(data, model, code_location):
    global dataset
    t1 = time()
    dataset = data
    #Find out whether the code is running on vm or my machine or the evaluators machine
    if (code_location == "vm"):
        load_vm()
    elif (code_location == "mach"):
        load_mach()
    elif (code_location == "eval"):
        load_eval()
    else:
        exit()

    #Run the required method
    if data == "forest":
        regression(FOREST_TEST, FOREST_TRAIN, model)
    elif data == "crime":
        regression(CRIME_TEST, CRIME_TRAIN, model)

    t2 = time()
    print 'Time taken in seconds: %f' % (t2 - t1)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])