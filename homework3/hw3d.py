
#Compute ALL predicted values for an RDD named rrd
predicted_values = final_rrd.map(lambda s: (s[1], np.dot(weights.T, s[2]), s[3]))

#Filter the en-data
en_train = final_rrd.filter(lambda s: (s[0] == 'en' and len(s[1])%2 ==0 ))
en_test = final_rrd.filter(lambda s: (s[0] == 'en' and len(s[1])%2 !=0 ))

#Get the feature matrix and sum up
mult = en_train.map(getFeatureMatrix)
feature_matrix = mult.reduce(lambda a,b:np.add(a,b))

#Get the xy product and sum up
mult = en_train.map(getFeatureTargetProduct)
xy_prod = mult.reduce(lambda a,b: np.add(a,b))

#Obtain the weights for linear regression
f_inv = np.linalg.inv(feature_matrix)
weights = np.dot(f_inv, xy_prod)

#Compute the predicted values for the test set
predicted_values = en_test.map(lambda s: (s[1], np.dot(weights.T, s[2]), s[3]))

#Filter out the yahoo-data
yahoo_predicted = predicted_values.filter(lambda s: s[0] == 'yahoo')
yahoo_predicted.first()