#__author__ = 'snehas'

from pyspark import SparkContext, SparkConf
import Iterator
import os
import numpy as np
import types


def getFeatureMatrix(x_array):
    x = np.matrix(x_array[2])
    xt = np.matrix.transpose(x)
    prod = np.multiply(x, xt)
    return prod

def getFeatureTargetProduct(x_array):
    x = x_array[2]
    y = x_array[3]
    prod = np.multiply(x, y)
    return prod

def get_val_list(values):
    val_list = list()
    if(values[0] is not None):
        if(isinstance(values[0], types.ListType)):
            val_list.append(values[0])
        elif(isinstance(values[0], types.TupleType)):
            val_list = val_list + get_val_list(values[0])
    if(values[1] is not None):
        if(isinstance(values[1], types.ListType)):
            val_list.append(values[1])
        elif(isinstance(values[1], types.TupleType)):
            val_list = val_list + get_val_list(values[1])
    return val_list


def getNumpyArray(s):
    arr = np.zeros(18)
    target = 0.0
    arr[0] = 1
    keys = s[0]
    values = s[1]
    val_list = get_val_list(values)
    for hour in val_list:
        if(int(hour[0]) == 23):
            target = int(hour[1])
        else:
            arr[int(hour[0]) - 5] = int(hour[1])
    return (keys[0], keys[1], arr, target);

conf = SparkConf().setAppName("app").setMaster("local[*]")
sc = SparkContext(conf=conf)


dirName = "/vagrant/shared_files/data/HW3Data/"
joined_rrd = sc.textFile(dirName + "6.txt")
joined_rrd = joined_rrd.map(lambda x: ((x.split()[0], x.split()[1]), ["6", x.split()[2]]))

for i in range(7, 24):
    filename = str(dirName)+ str(i) + ".txt"
    rrd = sc.textFile(filename)
    rrd = rrd.map(lambda x: ((x.split()[0], x.split()[1]), [str(i), x.split()[2]]))
    if(joined_rrd.count() < rrd.count()):
        joined_rrd = rrd.leftOuterJoin(joined_rrd)
    else:
        joined_rrd = joined_rrd.leftOuterJoin(rrd)
#     joined_rrd.count()
#     joined_rrd.take(5)
#
#
#
#     #rrd.count()
#     #rrd.take(1)
#
#     #rrdlist.append(rrd)
#
#     #rrdlist.append(sc.textFile(filename).map(lambda x: ((x.split()[0], x.split()[1]), [hourlist[i], x.split()[2]])))
#
#     if(i == 0):
#         joined_rrd = rrd
#     else:
#         if(joined_rrd.count() < rrd.count()):
#             joined_rrd = rrd.leftOuterJoin(joined_rrd)
#         else:
#             joined_rrd = joined_rrd.leftOuterJoin(rrd)
#
# joined_rrd = rrdlist[0]
#
# for i in range(1, len(rrdlist)):
#     if(joined_rrd.count() < rrdlist[i].count()):
#         joined_rrd = rrdlist[i].leftOuterJoin(joined_rrd)
#     else:
#         joined_rrd = joined_rrd.leftOuterJoin(rrdlist[i])

final_rrd = joined_rrd.map(getNumpyArray)
final_rrd.take(1)

