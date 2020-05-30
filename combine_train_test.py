# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 06:27:10 2020

@author: hi
"""

import sympy as sp
import numpy.matlib
import numpy as np
import xgboost
import matplotlib.pyplot as plt
import pandas as pd
import math
#from xgboost import XGBRegressor
#from xgboost import XGBClassifier
import datetime
import pickle as pkl
#import joblib
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import seaborn as sn
from sklearn.metrics import r2_score

a=datetime.datetime.now()
print(a)

z1 = np.load("train1gev.npy") #reading .npy file
z2 = np.load("train2gev.npy") #reading .npy file
z3 = np.load("train3gev.npy") #reading .npy file
z4 = np.load("train4gev.npy") #reading .npy file
z5 = np.load("train5gev.npy") #reading .npy file
z6 = np.load("train6gev.npy") #reading .npy file
z7 = np.load("train7gev.npy") #reading .npy file
z8 = np.load("train8gev.npy") #reading .npy file
z9 = np.load("train9gev.npy") #reading .npy file
z10 = np.load("train10gev.npy") #reading .npy file
b=datetime.datetime.now()
print(b)

print(b-a)

c=datetime.datetime.now()
print(c)

trainfile=np.row_stack((z1,z2,z3,z4,z5,z6,z7,z8,z9,z10))
np.save("train.npy",trainfile)   #saving the array direct in the .npy

d=datetime.datetime.now()
print(d)
print(d-c)
##############################################################################training

#flat_mat=np.load("train(1-10)gev.npy")
## split data into X(input  and y output)
#x_train = flat_mat[:,0:150528]
#y_train = flat_mat[:,150528]
#
#
#csr_xtrain=csr_matrix(x_train)
##print(csr_xtrain)
##csr_xtest=csr_matrix(X_test)
##print(csr_xtest)
#
## fit model no training data
#
#print("training starts")
#a=datetime.datetime.now()
#print(a)
#
#model = XGBRegressor(objective ="reg:squarederror", max_depth=7) # learning_rate=0.01, n_estimators=1, 
#model.fit(csr_xtrain, y_train)
#
#b=datetime.datetime.now()
#print(b)
#print("training ends")
#
#print("time taken for training", b-a) 
#
##Storing training fit parameters as pkl file
#model_pklFile=open("trained_model(1-5gev).pkl","wb")
#pkl.dump(model,model_pklFile)
#model_pklFile.close()  

####################testing#########################################################################

#dtest=np.load("test1.25gev.npy")
## split data into X(input  and y output)
#x_test = dtest[:,0:150528]
#y_test = dtest[:,150528]
#
###USING THE stored modle in pkl file
#mod=joblib.load("trained_model(1-5gev).pkl")
#
#
#csr_xtest=csr_matrix(x_test)
#print(csr_xtest)
#
#print("testing starts")
#c=datetime.datetime.now()
#print(c)
#
#y_pred = mod.predict(csr_xtest)
##predictions = [round(value) for value in y_pred]
#
#d=datetime.datetime.now()
#print(d)
#print("testing ends")
#print("time taken for testing", d-c)
##
##yp = np.round(y_pred,2)
##accuracy = accuracy_score(y_test, yp)
#accuracy =r2_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#print(accuracy)
##
#plt.scatter(y_test, y_pred) #label=y) #for scatter plot of predicted and actual y values third term is to save accuracy
#sn.set_style("white")
#plt.xlabel("expected values")#y_test value
#plt.ylabel("predicted values")#y_pred value
##plt.legend()  #to get the y on the top of graph
#plt.savefig("y1_test(x)y_predicted.png")
#plt.show()
#
#from sklearn.metrics import mean_squared_error
#r=mean_squared_error(y_test, y_pred)
#print(r*100)
#plt.hist(y_pred-y_test)
#plt.show()
