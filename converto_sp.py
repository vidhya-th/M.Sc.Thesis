 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 08:37:42 2020

@author: lakshmi
"""
#for 1 event
#convert_tostrip(spy_mat).py
#Displaying combin&_train&_test.py.

import sympy as sp
import numpy.matlib
import numpy as np

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


a=datetime.datetime.now()
print(a)

df = pd.read_csv("10GeVm-dwn100eve.txt", sep='\t')
df1=np.array(df)

d1=df.drop(['posy','pid', 'detectorId', 'is_primary', 'del_posx', 'del_posy', 'del_posz', 'del_t', 'globTime', 'propTime', 'localTime', 'ni_edep','tot_edep','tot_E', 'tot_KE','px','py','pz', ], axis=1)
d1=np.array(d1)
#nlayers = 60
nstrips = 512 #16m*32
nlayers=147
eid=d1[:,0]
m=int(max(eid))
m1=min(eid)
sx=[]
sy=[]
print(m)
#nrows = d1.shape[0]
#print(nrows)

flat_mat=[]
train_m=[]

#f_mat= np.zeros ((, 150529))
for i in range (m+1):
    print("EVENT",i)
    eid_ds = df[df['eid']==i]
    eid_a=np.array(eid_ds)
    nrows = eid_a.shape[0]
    x_pos=[]
    y_pos=[]
    
    for j in range(0,nrows):
        x_pos.append(eid_a[j,6])
        
        y_pos.append(eid_a[j,4])    

    
    matx = np.zeros ((nlayers, nstrips))
    for k in range (73, nlayers):
        if(k-73<=len(x_pos)-1):            
            stripx= int ((x_pos[k-73]+ 8000) * nstrips / 16000)            
            if (stripx >0 and stripx <=511):
                matx[k, stripx] = 1
    plt.spy(matx, origin = 'lower', aspect=5, markersize=4, color='black')
    plt.show()

    maty = np.zeros ((nlayers, nstrips))
    for l in range (73, nlayers):
        if(l-73<=len(y_pos)-1):            
            stripy= int ((y_pos[l-73]+ 8000) * nstrips / 16000)            
            if (stripy >0 and stripy <=511):
                maty[l, stripy] = 1
#                print("l",l,"stripy", stripy)
#    plt.spy(maty, origin = 'lower', aspect=5, markersize=4, color='black')
#    
#    plt.show()
#   
#    momt=[math.sqrt(eid_a[0,18]**2+eid_a[0,19]**2+eid_a[0,20]**2)]
    eng=4
    rxmat=np.reshape(matx,(1,np.product(matx.shape)))
#    print(np.count_nonzero(rxmat))

    rymat=np.reshape(maty,(1,np.product(maty.shape)))
#    print(np.count_nonzero(rymat))

    train_mat=np.column_stack((rxmat,rymat,eng))
    train_m.append(train_mat)
    
flat_mat=np.asarray(train_m)
#print(np.shape(flat_mat))
flat_mat1=numpy.reshape(flat_mat, (100, 150529))
print(np.count_nonzero(flat_mat))
#
#
np.save("train10gev.npy",flat_mat1)   #saving the array direct in the .npy
#z = np.load("test_npy_file.npy") #reading .npy file

#########################################################################################################################################################################################

    
    




 



