
#Compute RMSE
count = predicted_values.count()
rmse_square = predicted_values.map(lambda s: (s[1] - s[2])*(s[1]-s[2]))
rmse_square = rmse_square.reduce(lambda a,b:a+b)
rmse_square = rmse_square/count
rmse_square = math.sqrt(rmse_square)
print "RMSE: " + str(rmse_square)