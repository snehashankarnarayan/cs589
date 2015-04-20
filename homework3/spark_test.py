#__author__ = 'snehas'

from pyspark import SparkContext, SparkConf
import os
import numpy as np
import types
import math
import sys
from time import time

#Spark Context
conf = SparkConf().setAppName("app").setMaster("local[*]")
sc = SparkContext(conf=conf)