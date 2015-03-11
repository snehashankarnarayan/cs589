#!/usr/bin/python

from regression import run
import sys
import os

if __name__ == "__main__":
    if(len(sys.argv) <= 1):
        print "Specify command line arg: dataset (crime/forest). Use ./run_me.py crime"
        exit()
    directory = "output/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    dataset = sys.argv[1]
    if(dataset == "crime"):
        run("crime", "knn", "eval" )
    elif(dataset == "forest"):
        run("forest","svr", "eval")

