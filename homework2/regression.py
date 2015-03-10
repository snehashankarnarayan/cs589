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
from sklearn.metrics import *
from sklearn.ensemble import *

# Dataset locations
CRIME_TEST = ""
CRIME_TRAIN = ""
FOREST_TEST = ""
FOREST_TRAIN = ""

dataset = ""

#Code to select hyper parameters using cross validation




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

#Best subset selection
def bestSubsetSelection(regress, data, y_class):
    min_score = 99999
    num_predictors = len(data[0]);
    subset = []
    scores = dict()
    for k in range(0,num_predictors ):
        print "Doing: " + str(k)
        arr = itertools.combinations(range(num_predictors), k)
        combinations = list(arr)
        for item in combinations:
            print item
            x_data = np.delete(data, item, 1)
            score = -cross_val_score(regress, x_data, y_class, cv = 5, scoring='mean_squared_error').mean()
            scores[item] = score
            if(score < min_score):
                min_score = score
                subset = item

    pprint(scores)
    print min_score, subset
    return subset

def get_hyperparams_svr(x_data, y_class):
    scores = dict()
    kernel_list = ['rbf']
    gamma_list = [0.001, 0, 0.01, 0.1, 0.5, 1, 2, 4]
    epsilon_list = [0, 0.01, 0.1, 0.5, 1, 2, 4]
    C_list = [0.5, 1, 2, 4, 10, 25, 50, 100, 200]

    min_score = 9999.00
    max_kernel = ""
    max_gamma = -1.0
    max_epsilon = -1.0
    max_C = -1.0


    for kernel in kernel_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                for C in C_list:
                    clf = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)
                    print "Doing " + kernel + ", " + str(gamma) + ", " + str(C) + ", " + str(epsilon)
                    score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_absolute_error').mean()
                    scores[kernel,gamma,epsilon,C] = score
                    if(score < min_score):
                        min_score = score
                        max_kernel = kernel
                        max_epsilon = epsilon
                        max_C = C
                        max_gamma = gamma

    print "Table"
    for kernel in kernel_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                for C in C_list:
                    print kernel + " & " + str(gamma) + " & " + str(epsilon) + " & " + str(C) + " & " + str(scores[kernel,gamma,epsilon,C]) + " \\\\"


    print max_C, max_gamma, max_kernel, max_epsilon
    return max_C, max_gamma, max_kernel, max_epsilon

def get_hyperparams_knn(x_data, y_class):
    scores = dict()
    k_list = range(1,10)
    distance_list = ['manhattan', 'euclidean', 'chebyshev']
    weight_list = ['uniform', 'distance']

    min_score = 9999.00
    max_k = 0
    max_distance = ""
    max_weight = ""

    for k in k_list:
        for distance in distance_list:
            for weight in weight_list:
                clf = neighbors.KNeighborsRegressor(n_neighbors = k, weights = weight, metric = distance)
                #print "Doing " + distance + ", " + str(k) + ", " + weight
                score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_squared_error').mean()
                scores[k, distance, weight] = score
                if(score < min_score):
                    min_score = score
                    max_k = k
                    max_distance = distance
                    max_weight = weight

    print "Table"
    for k in k_list:
        for distance in distance_list:
            for weight in weight_list:
                print str(k) + " & " + distance + " & " + weight + " & " + str(scores[k,distance,weight]) + " \\\\"

    print "Graph"
    for distance in distance_list:
        for weight in weight_list:
            for k in k_list:
                print str(k) + " " + distance + " " + weight + " " + str(scores[k,distance,weight])

    print max_k, max_distance, max_weight
    return max_k, max_distance, max_weight


def inbuiltSubsetSelection(regress, data, y_class):
    print "Inbuilt subset feature selection"
    min_score = 9999
    num_predictors = len(data[0]);
    scores = dict()
    for k in range(1, num_predictors ):
        print "Doing: " + str(k)
        x_data = SelectKBest(f_regression,k = k).fit_transform(data,y_class)
        score = -cross_val_score(regress, x_data, y_class, scoring='mean_absolute_error').mean()
        scores[k] = score
        if(score < min_score):
            min_score = score
            subset_data = x_data

    for k in range(0, len(data[0])):
        print str(k) + " " + str(scores[k])

    print min_score
    return subset_data


def anti_transform(data):
    for i in range(0, len(data)):
        data[i] = math.exp(data[i]) - 0.0001
    return data

def transform(data):
    for i in range(0, len(data)):
        data[i] = math.log(data[i] + 0.0001)
    return data

def get_score(regress, x_data, y_class):
    regress.fit(x_data[:int(.85*len(x_data)),:], y_class[:int(.85*len(x_data))])
    return regress.score(x_data[int(.85*len(x_data)+1):,:], y_class[int(.85*len(x_data)) + 1 :])

#Backward stepwise selection
def backwardStepwiseSelection(regress, data, y_class):
    print "Backward stepwise selection"
    min_score = 9999
    max_k = -1
    scores = dict()
    for k in range(0, len(data[0])):
        print "Doing: " + str(k)
        x_data = np.delete(data, k, 1)
        score = -cross_val_score(regress, x_data, y_class, scoring='mean_squared_error').mean()
        scores[k] = score
        if(score < min_score):
            min_score = score
            max_k = k

    for k in range(0, len(data[0])):
        print str(k) + " " + str(scores[k])

    print min_score, max_k
    return max_k

#Regression function
def regression(test_location, train_location, regression_method):
    train = np.load(train_location)
    
    #Get the required training data
    x_data = train[:, 1:]  #All but the first column
    y_class = train[:, 0]

    y_class = transform(y_class)

    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    #SVR
    if (regression_method == "svr"):
        #C, gamma, kernel, epsilon = get_hyperparams_svr(x_data, y_class)
        #regress = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)
        regress = SVR()
        #regress = SVR(kernel='rbf', C = 0.5, gamma = 0.001, epsilon=0.01)


    if ( regression_method == "knn"):
        #k, distance, weight = get_hyperparams_knn(x_data, y_class)
        #regress = neighbors.KNeighborsRegressor(n_neighbors = k, weights = weight, metric = distance)
        regress = neighbors.KNeighborsRegressor()

    elif (regression_method == "lin"):
        regress = LassoLarsCV()

    elif (regression_method == "tree"):
        regress = DecisionTreeRegressor()

    elif (regression_method == "gauss"):
        regress = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=100)


    #Place holder for any random classifier I want to try from sklearn.ensemble
    elif (regression_method == "rand"):
        regress = ensemble.BaggingRegressor(base_estimator = SVR(kernel='rbf', C = 25, gamma = 4, epsilon=4), n_estimators = 20)


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

    out = anti_transform(out)
    print 'Prediction time taken in seconds: %f' % (t4 - t3)

    #Spit the output into CSV file
    output(out, regression_method)

def run(data, code_location, model):
    global dataset
    t1 = time()
    dataset = data
    #Find out whether the code is running on vm or my machine or the evaluators machine
    if (model == "vm"):
        load_vm()
    elif (model == "mach"):
        load_mach()
    elif (model == "eval"):
        load_eval()
    else:
        exit()

    #Run the required classification
    if data == "forest":
        regression(FOREST_TEST, FOREST_TRAIN, code_location)
    elif data == "crime":
        regression(CRIME_TEST, CRIME_TRAIN, code_location)

    t2 = time()
    print 'Time taken in seconds: %f' % (t2 - t1)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])