#!/usr/bin/python

from regression import run
import os

#Runs the pipeline for both datasets and spits out the CSV files
if __name__ == "__main__":
    directory = "output/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    run("crime", "crime_pipe", "eval" )
    run("forest","forest_pipe", "eval")

