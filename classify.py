#!/usr/bin/python

import numpy as np
import sys
from time import time
import csv
from pprint import pprint
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.externals.six import StringIO
from subprocess import call

from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn import linear_model

#Dataset locations
BEATS_TEST = ""
BEATS_TRAIN = ""
SIGN_TEST = ""
SIGN_TRAIN = ""
SPAM_TEST = ""
SPAM_TRAIN = ""
BLACKBOX_TEST = ""
BEATS_TRAIN = ""
dataset = ""

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

def show_output(data, classifier_name):
    print classifier_name
    pprint(data)
    for d in data:
        print d

def output(data, classifier_name):
    #show_output(data, classifier_name)
    with open("output/" + dataset + "_" + classifier_name + ".csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID','Category'])
        for i in range(0, len(data)):
            writer.writerow([i+1, int(data[i])])

#Classify function
def classify(test_location, train_location, classifier_name):
    train = np.load(train_location)
    #Get the required training data
    x_data = train[:,1:] #All but the first column
    y_class = train[:, 0]

    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    #SVM
    if(classifier_name == "svm"):
        clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
          gamma=0.001, max_iter=-1, probability=False,
          random_state=None, shrinking=True, tol=0.001, verbose=False);

    #KNN Classifier
    elif(classifier_name == "knn"):
        clf = neighbors.KNeighborsClassifier()

    #LDA
    elif(classifier_name == "lda"):
        clf = LDA()

    #Naive Bayes
    elif(classifier_name == "nb"):
        clf = GaussianNB()

    #LogisticRegression
    elif(classifier_name == "lr"):
        clf = linear_model.LogisticRegression(C=1000)

    clf.fit(x_data, y_class)
    output(clf.predict(test_data), classifier_name)

if __name__ == "__main__":
    global dataset
    t1 = time()
    dataset = sys.argv[1]
    if(sys.argv[3] == "vm"):
        load_vm()
    elif(sys.argv[3] == "mach"):
        load_mach()
    else:
        exit()

    if sys.argv[1] == "beats":
        classify(BEATS_TEST, BEATS_TRAIN, sys.argv[2])
    elif sys.argv[1] == "spam":
        classify(SPAM_TEST, SPAM_TRAIN, sys.argv[2])
    elif sys.argv[1] == "sign":
        classify(SIGN_TEST, SIGN_TRAIN, sys.argv[2])
    elif sys.argv[1] == "blackbox":
        classify(BLACKBOX_TEST, BLACKBOX_TRAIN, sys.argv[2])
    t2 = time()
    print 'Time taken in seconds: %f' %(t2-t1)


