#!/usr/bin/python

import numpy as np
import sys
from time import time
import csv
from pprint import pprint
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn import linear_model
from sklearn import ensemble

# Dataset locations
BEATS_TEST = ""
BEATS_TRAIN = ""
SIGN_TEST = ""
SIGN_TRAIN = ""
SPAM_TEST = ""
SPAM_TRAIN = ""
BLACKBOX_TEST = ""
BEATS_TRAIN = ""
dataset = ""

#Code to select hyperparameters using training accuracy
def select_hyperparams_train_accuracy(x_data, y_class):
    #Range of hyperparameters to be tested over
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metric_list = ["manhattan", "euclidean", "chebyshev"]

    #Have a dictionary for keeping track of validation error
    error_sums = dict()

    for k in k_list:
        for metric in metric_list:
            clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(x_data, y_class)
            vals = clf.predict(x_data)
            error = 0
            for z in range(0, len(vals)):
                #Getting the number of wrong predictions
                a = 0
                b = 0
                a = a + vals[z]
                b = b + y_class[z]
                if (a != b):
                    error = error + 1
            error = error * 100 / len(vals)
            error_sums[k, metric] = error

    #Getting the final error percentages and minimum
    min = sys.maxint
    final_k = 0
    final_metric = ""
    for k in k_list:
        for metric in metric_list:
            if (error_sums[k, metric] < min):
                min = error_sums[k, metric]
                final_k = k
                final_metric = metric

    pprint(error_sums)
    print final_k, final_metric

    #Returning the final k value and final distance metric for KNN
    return final_k, final_metric

#Code to select hyper parameters using cross validation
def select_hyperparams_cross_validation(init_train):
    #Randomly partition training data into 80:20 for training:test data for K fold cross validation
    np.random.shuffle(init_train)
    partition_index = int(0.8 * len(init_train))
    training, test = init_train[:partition_index], init_train[partition_index:]

    #Range of hyperparameters to be tested over
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metric_list = ["manhattan", "euclidean", "chebyshev"]

    #Have a dictionary for keeping track of validation error
    error_sums = dict()
    scores = dict()

    #Implementing k fold with k = 10
    K = 10
    partition_index = int(len(training) / K)
    prev = 0
    for q in range(1, K + 1):
        #One block is used for validation
        v = training[prev: q * partition_index]

        #Getting the rest of the K - 1 blocks
        if ( q == 1):
            tr = training[q * partition_index:]
        else:
            tr = training[0: prev]
            for x in range(q * partition_index, len(training)):
                tr = np.vstack([tr, training[x]])

        #Separating the data vectors, labels for the training and test set for this fold
        tr_data = tr[:, 1:]
        tr_class = tr[:, 0]
        v_data = v[:, 1:]
        v_class = v[:, 0]

        #Train for each set of hyperparameters
        for k in k_list:
            for metric in metric_list:
                clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
                clf.fit(tr_data, tr_class)
                vals = clf.predict(v_data)
                scores[k, metric] = clf.score(v_data, v_class)
                error = 0
                for z in range(0, len(vals)):
                    #Getting the number of wrong predictions
                    a = 0
                    b = 0
                    a = a + vals[z]
                    b = b + v_class[z]
                    if (a != b):
                        error = error + 1
                error = error * 100 / len(v_class)

                #Summing up the errors per hyperparameter set
                t = tuple([k, metric])
                if (error_sums.get(t) != None):
                    error_sums[k, metric] = error_sums[k, metric] + error
                else:
                    error_sums[k, metric] = error
        prev = q * partition_index

    #Getting the final error percentages and minimum
    min = sys.maxint
    final_k = 0
    final_metric = ""
    for k in k_list:
        for metric in metric_list:
            error_sums[k, metric] = error_sums[k, metric] * 1.0 / K
            if (error_sums[k, metric] < min):
                min = error_sums[k, metric]
                final_k = k
                final_metric = metric

    for k in k_list:
        for metric in metric_list:
            print str(k) + " & " + metric + " & " + str(error_sums[k, metric]) + " & " + str(scores[k, metric]) + " \\\\";

    for k in k_list:
        print str(k) + " " + str(error_sums[k, "euclidean"])
    for k in k_list:
        print str(k) + " " + str(error_sums[k, "manhattan"])
    for k in k_list:
        print str(k) + " " + str(error_sums[k, "chebyshev"])
    print final_k, final_metric

    #Returning the final k value and final distance metric for KNN
    return final_k, final_metric


#Function that stores the paths of the dataset in my machine
def load_mach():
    global BEATS_TEST, BEATS_TRAIN, BLACKBOX_TEST, BLACKBOX_TRAIN, SPAM_TRAIN, SPAM_TEST, SIGN_TRAIN, SIGN_TEST
    BEATS_TEST = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/Beats/test_distribute.npy'
    BEATS_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/Beats/train.npy'

    SIGN_TEST = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/SignLanguage/test_distribute.npy'
    SIGN_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/SignLanguage/train.npy'

    SPAM_TEST = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/Spam/test_distribute.npy'
    SPAM_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/Spam/train.npy'

    BLACKBOX_TEST = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/BlackBox/test_distribute.npy'
    BLACKBOX_TRAIN = '/Users/snehas/vagroot/shared_files/data/HW1DataDistribute/BlackBox/train.npy'


#Function that stores the paths of the dataset in my VM
def load_vm():
    global BEATS_TEST, BEATS_TRAIN, BLACKBOX_TEST, BLACKBOX_TRAIN, SPAM_TRAIN, SPAM_TEST, SIGN_TRAIN, SIGN_TEST
    BEATS_TEST = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/Beats/test_distribute.npy'
    BEATS_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/Beats/train.npy'

    SIGN_TEST = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/SignLanguage/test_distribute.npy'
    SIGN_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/SignLanguage/train.npy'

    SPAM_TEST = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/Spam/test_distribute.npy'
    SPAM_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/Spam/train.npy'

    BLACKBOX_TEST = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/BlackBox/test_distribute.npy'
    BLACKBOX_TRAIN = '/home/vagrant/Desktop/shared_files/data/HW1DataDistribute/BlackBox/train.npy'

def load_eval():
    global BEATS_TEST, BEATS_TRAIN, BLACKBOX_TEST, BLACKBOX_TRAIN, SPAM_TRAIN, SPAM_TEST, SIGN_TRAIN, SIGN_TEST
    BEATS_TEST = '/vagrant/shared_files/data/HW1DataDistribute/Beats/test_distribute.npy'
    BEATS_TRAIN = '/vagrant/shared_files/data/HW1DataDistribute/Beats/train.npy'

    SIGN_TEST = '/vagrant/shared_files/data/HW1DataDistribute/SignLanguage/test_distribute.npy'
    SIGN_TRAIN = '/vagrant/shared_files/data/HW1DataDistribute/SignLanguage/train.npy'

    SPAM_TEST = '/vagrant/shared_files/data/HW1DataDistribute/Spam/test_distribute.npy'
    SPAM_TRAIN = '/vagrant/shared_files/data/HW1DataDistribute/Spam/train.npy'

    BLACKBOX_TEST = '/vagrant/shared_files/data/HW1DataDistribute/BlackBox/test_distribute.npy'
    BLACKBOX_TRAIN = '/vagrant/shared_files/data/HW1DataDistribute/BlackBox/train.npy'


#Function to spit the output in an appropriately named CSV file
def output(data, classifier_name):
    with open("output/" + dataset + "_" + classifier_name + ".csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'Category'])
        for i in range(0, len(data)):
            writer.writerow([i + 1, int(data[i])])


#Classify function
def classify(test_location, train_location, classifier_name):
    train = np.load(train_location)
    #Get the required training data
    x_data = train[:, 1:]  #All but the first column
    y_class = train[:, 0]

    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    print 'Done loading data'

    #SVM
    if (classifier_name == "svm"):
        clf = svm.SVC();

    #Decision tree
    elif (classifier_name == "tree"):
        clf = tree.DecisionTreeClassifier()

    #KNN Classifier
    elif (classifier_name == "knn"):
        #k, metric = select_hyperparams_cross_validation(train)
        #k, metric = select_hyperparams_train_accuracy(x_data, y_class)
        k , metric = 5, "minkowski"
        clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)

    #LDA
    elif (classifier_name == "lda"):
        clf = LDA()

    #Naive Bayes
    elif (classifier_name == "nb"):
        clf = GaussianNB()

    #LogisticRegression
    elif (classifier_name == "lr"):
        clf = linear_model.LogisticRegression()

    #Place holder for any random classifier I want to try from sklearn.ensemble
    elif (classifier_name == "rand"):
        clf = ensemble.BaggingClassifier(base_estimator=neighbors.KNeighborsClassifier(n_neighbors=4), n_estimators=23)

    #Train the classifier and compute time taken
    t1 = time()
    clf.fit(x_data, y_class)
    t2 = time()

    print 'Training time taken in seconds: %f' % (t2 - t1)

    #Do the predictions and compute time taken
    t3 = time()
    out = clf.predict(test_data)
    t4 = time()

    print 'Prediction time taken in seconds: %f' % (t4 - t3)

    #Spit the output into CSV file
    output(out, classifier_name)


if __name__ == "__main__":
    global dataset
    t1 = time()
    dataset = sys.argv[1]
    #Find out whether the code is running on vm or my machine
    if (sys.argv[3] == "vm"):
        load_vm()
    elif (sys.argv[3] == "mach"):
        load_mach()
    elif (sys.argv[3] == "eval"):
        load_eval()
    else:
        exit()

    #Run the required classification
    if sys.argv[1] == "beats":
        classify(BEATS_TEST, BEATS_TRAIN, sys.argv[2])
    elif sys.argv[1] == "spam":
        classify(SPAM_TEST, SPAM_TRAIN, sys.argv[2])
    elif sys.argv[1] == "sign":
        classify(SIGN_TEST, SIGN_TRAIN, sys.argv[2])
    elif sys.argv[1] == "blackbox":
        classify(BLACKBOX_TEST, BLACKBOX_TRAIN, sys.argv[2])
    t2 = time()
    print 'Time taken in seconds: %f' % (t2 - t1)


