#__author__ = 'snehas'

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("app").setMaster("local[*]")
sc = SparkContext(conf=conf)

lines = sc.textFile("/vagrant/shared_files/data/HW3Data/6.txt")
print lines.count()