
#This function computes the feature target product from each record in rdd
def getFeatureTargetProduct(x_array):
    x = x_array[2]
    y = x_array[3]
    prod = np.multiply(x, y)
    return prod

#Get the xy product and sum up
mult = final_rrd.map(getFeatureTargetProduct)
xy_prod = mult.reduce(lambda a,b: np.add(a,b))