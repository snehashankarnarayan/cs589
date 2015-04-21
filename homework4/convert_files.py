#!/usr/bin/python


import glob
import os

for file in glob.glob("*.eps"):
    print "Converting" + file
    os.system('epstopdf ' + file)