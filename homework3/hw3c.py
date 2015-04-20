
#Obtain the weights for linear regression
f_inv = np.linalg.inv(feature_matrix)
weights = np.dot(f_inv, xy_prod)