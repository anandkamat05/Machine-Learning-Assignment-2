from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import string as str
import re
import random
import numpy as np
from mpmath.functions.expintegrals import square_exp_arg


####################################### Import Data #################################################
path = "C:/Users/anand/OneDrive/University McGill/Machine Learning/Assignment2/hw2pca.txt"

data = []
with open(path, 'rb') as text_file:
    for x in text_file:
        data.append(x)

for i in range(0,len(data)):
    values = str.split(data[i], sep = '\t')
    values = map(str.lstrip,values)
    values = map(float, values[:len(values)-1])
    data[i] = values
    
################################### Shuffle Data ######################################################
random.shuffle(data)
 
# ################################### Create Data Set and Center it ###############################################
 
X_train = data[:int((0.8)*len(data)-1)]
X_test = data[int((0.8)*len(data)-1):]

#Center Data
X_train_mean = np.mean(X_train)
X_train -= X_train_mean
X_test_mean = np.mean(X_test)
X_test -= X_test_mean

# ################################################# PCA ALGORITHM on Training and Test Data #####################################

recontruction_train_error = []
train_variance = []
recontruction_test_error = []
test_variance = []

for i in range(0, len(X_train[0])):
    pca = PCA(n_components= i+1)
    clf = pca.fit(X_train)
    transformed_train_data = clf.transform(X_train)
    transformed_test_data = clf.transform(X_test)
    
    recontructed_train_data = pca.inverse_transform(transformed_train_data)
    recontructed_test_data = pca.inverse_transform(transformed_test_data)

    r_train_error = 0
    r_test_error = 0
    
    for d in range(0, len(X_train)):
        for e in range(0, len(X_train[0])):
            r_train_error += ((abs(X_train[d][e] - recontructed_train_data[d][e]))**2)
    recontruction_train_error.append(r_train_error)
    
    for f in range(0, len(X_test)):
        for g in range(0, len(X_test[0])):
            r_test_error += ((abs(X_test[f][g] - recontructed_test_data[f][g]))**2)
    recontruction_test_error.append(r_test_error)
    
    
############################################# Calculating Variance for Data #######################
pca2 = PCA(n_components= 250)
clf2 = pca.fit(X_train)

var_train = clf2.explained_variance_

############################################## Plotting The Reconstruction Error Graphs ###################################    
plt.subplot(211)
plt.xlabel("Dimensions")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error as a function of Dimensions")

train_dimensions = [(x+1) for x in range(0,len(X_train[0]))]
line1, = plt.plot(train_dimensions, recontruction_train_error, label = "Training Data")

test_dimensions = [(x+1) for x in range(0,len(X_test[0]))]
line2, = plt.plot(test_dimensions, recontruction_test_error, label = "Test Data")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

############################################## Plotting The Variance Graphs ###################################    
plt.subplot(212)
plt.xlabel("Dimensions")
plt.ylabel("Variance")
plt.title("Variance as a function of Dimensions")
 
train_dimensions = [(x+1) for x in range(0,len(X_train))]
line3, = plt.plot(train_dimensions, var_train)

plt.show()
