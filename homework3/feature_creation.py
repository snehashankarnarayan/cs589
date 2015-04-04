#__author__ = 'snehas'

from pyspark import SparkContext, SparkConf
import os
import numpy
import types

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
    arr = numpy.zeros(18)
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
    return (keys, arr, target);


conf = SparkConf().setAppName("app").setMaster("local[*]")
sc = SparkContext(conf=conf)

dirName = "/vagrant/shared_files/data/HW3Data/"
files = os.listdir(dirName)
rrdlist = list()
hourlist = list()

for file in files:
    hour = file.split('.')[0]
    hourlist.append(hour)

for i in range(0, len(files)):
    filename = str(dirName)+str(files[i])
    rrd = sc.textFile(filename).map(lambda x: ((x.split()[0], x.split()[1]), [hourlist[i], x.split()[2]]))
    # Make sure hour is augmented properly
    rrdlist.append(rrd)

joined_rrd = rrdlist[0]

for i in range(1, len(rrdlist)):
    joined_rrd = joined_rrd.union(rrdlist[i])
    if(joined_rrd.count() < rrdlist[i].count()):
        joined_rrd = rrdlist[i].leftOuterJoin(joined_rrd)
    else:
        joined_rrd = joined_rrd.leftOuterJoin(rrdlist[i])

final_rrd = joined_rrd.map(getNumpyArray)
final_rrd.take(0)

