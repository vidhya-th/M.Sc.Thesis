# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:04:47 2020

@author: hi
"""
import sympy as sp
import numpy.matlib
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd

import math
from pprint import pprint
from numpy.linalg import inv
from scipy import interpolate
from scipy import constants
plt.style.use("ggplot")

#DATAFRAMES
df = pd.read_csv("5GeV_1000eve.txt",sep="\t")
me = pd.read_csv("mu-iron-energyloss.csv")
# f_size=len(df)
# f_size=3

#ARRAY FROM DATAFRAMES
df1=np.array(df)
# ke=df['tot_KE']*10**3  #MeV
mom=me['p']
moml=mom.tolist()
de_dx=me['dE/dx']
eid=df1[:,0]
en=max(eid)
print("No. of events:",(en+1))



#CONSTANTS
c=constants.c
e=constants.e
Bx = 1.5                     #T
By = 0.0
Bz = 0.0
step_size = 56.2               #mm
# qp=3.6697713422062606e-23
# qp =e*c/(5*10**3)
# print("qpInitial",qp)
k_h  = 299790               # MeV/c/T/mm
mass_mu = 105.6583755        # Mev/c^2
mass_e = 0.511               # MeV/c^2
rl_fe = 1.757                # cm    radiation length
A_fe = 55.845                # mass number
Z_fe = 26                    # atomic number
d_fe = 5.6                   # cm thickness of fe
rho_fe = 7.874               # g/cc
en_loss_fe= 1.594            # Mev c^2/g0
beta = 1
gamma = 10
f_size =0

#plt.scatter(df['posy'],df['momy'],df['momz'])

#LISTS
QP = []
# QP = [3.2140831407612124e-23]
# QP =[e/(mass_mu*c)]

QP_S=[]
P_S=[]



snx=[]          #strip numbers along x
sny=[]          #strip numbers along y

Ck_matrix=[]
F_matrix=[]
x_k1_fwd=[]
Qk_matrix=[]
c_k1_step=[[10**6,0,0,0,0],[0,10**6,0,0,0],[0,0,10**6,0,0],[0,0,0,10**6,0],[0,0,0,0,10**6]]
x_k1_step=[]
F_r_a=[]
FNF=[]
Ckrr=[]
diff_eve=[]
plot_pf=[]
plot_ps=[]
filter_state_vector=np.empty([5,1], dtype='float')
nstrips=512   #16*32
mu_event=[]
sigma_event=[]

def measurements():
    #the coordinated are transformed to match with KB's Kalman filter
    y = df['posx']
    z = df['posy']
    x = df['posz']
    return x,y,z

def measurement(a):
    x,y,z = measurements()
    return x[a], y[a], z[a]

def print_slopes():
    x,y,z = measurements()
    xx =[]
    sx =[]
    sy =[]
    for i in range(meas_size()-1):
        slpx=(x[i+1]-x[i])/(z[i+1]-z[i])
        slpy=(y[i+1]-y[i])/(z[i+1]-z[i])
        print(i,slpx,slpy)
        xx.append(i)
        sx.append(slpx)
        sy.append(slpy)
    plt.plot(xx,sx)
    plt.plot(xx,sy)

def meas_size():
    return len(df['posx'])

def Sx(dz):
    return 0.5*Bx*dz**2

def Rx(dz):
    return Bx*dz

def Sy(dz):
    return 0.5*By*dz**2

def Ry(dz):
    return By*dz

def Sxx(dz):
    return (Bx**2*dz**3)/6

def Rxx(dz):
    return (Bx**2*dz**2)/2

def Sxy(dz):
    return (Bx*By*dz**3)/6

def Rxy(dz):
    return (Bx*By*dz**2)/2

def Syx(dz):
    return (Bx*By*dz**3)/6

def Ryx(dz):
    return (Bx*By*dz**2)/2

def Syy(dz):
    return (By**2*dz**3)/6

def Ryy(dz):
    return (By**2*dz**2)/2

def h(tx,ty,qP):
    return k_h*qP*math.sqrt(1+tx**2+ty**2)

def predict_x(x0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return x0 + tx*dz + hh*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ hh**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))

def predict_y(y0,tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return y0 + ty*dz + hh*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))+ hh**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))

def predict_tx(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return  tx + hh*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))+ hh**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))

# def predict_txdummy(tx,ty,qP,dz):
#     hh = h(tx,ty,qP)
#     return  tx + hh*(tx*ty*Rx(dz))+ hh**2*(tx*(3*ty*2+1)*Rxx(dz) )

def predict_ty(tx,ty,qP,dz):
    hh = h(tx,ty,qP)
    return ty + hh*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ hh**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))

# def predict_tydummy(tx,ty,qP,dz):
#     hh = h(tx,ty,qP)
#     return ty + hh*((1+ty**2)*Rx(dz))+ hh**2*(ty*(3*ty*2+3)*Rxx(dz))    


######################## Event Loop ########################

for n in range(en+1):
    eid_ds = df[df['eid']==n]
    f_size=len(eid_ds)      # is the number of the layers per event
    eid_a=np.array(eid_ds)  # array of dataframe pertaining to one particular event
    # print("eid",eid_a[0,0],"layers:",f_size)
    print("@@@@@@@\nEVENT ",n,"BEGINS @@@@@@")
#    print("No.of layers",f_size)
    
    
    xf_fwd= yf_fwd=txf_fwd=tyf_fwd=qpf_fwd=0
  

    xf1=[]
    yf1=[]
    txf1=[]
    tyf1=[]
    qpf1=[]

    xp=[]
    yp=[]
    txp=[]
    typ=[]
    qpp=[]
    
    P2=[]
    
    xm=[]
    ym=[]
    txm=[]
    tym=[]
    P_F=[5*10**3]  #MeV/c
    P_F_lyr=[5*10**3]
    E_inc=[5*10**3]
    E_atplane=[]
    E_atplane_S=[]
    FN = np.matlib.eye(n = 5, M = 5, k = 0, dtype = float)
    FNF=[]
    
    
    #####################   Layer Loop  #######################################
    for i in range(f_size-1):
        # print("-----------------")
        # print("Layer ",i,"to",i+1)
        # print("-----------------")
        
        
        #############    Measurement through strip digitization   ###################
        stripx= int ((eid_a[i,6]+ 8000) * nstrips / 16000)   
        snx.append(stripx)
        stripy= int ((eid_a[i,4]+ 8000) * nstrips / 16000) 
        sny.append(stripy)
        # print("StripX:",stripx)
        # print("StripY:",stripy)
       

        xp_step=[]
        yp_step=[]
        txp_step=[]
        typ_step=[]
        qpp_step=[]

        if(i==0):
            # x0_lyr=snx[i]
            # y0_lyr=sny[i]
            x0_lyr=eid_a[i,6]
            y0_lyr=eid_a[i,4]
            tx0_lyr=(eid_a[i+1,6]-eid_a[i,6])/(eid_a[i+1,5]-eid_a[i,5])
            ty0_lyr=(eid_a[i+1,4]-eid_a[i,4])/(eid_a[i+1,5]-eid_a[i,5])
            qp0_lyr=0

        else:
            #filter
            x0_lyr=xf_fwd
            y0_lyr=yf_fwd
            tx0_lyr=txf_fwd
            ty0_lyr=tyf_fwd
            qp0_lyr=qpf_fwd

        
        # print("Layer-wise State parameters :",x0_lyr,y0_lyr,tx0_lyr,ty0_lyr)

        def predict_qp_f():
            x1=[]
            y1=[]
            index=0
            for j in mom:
            #print(j)
                if((P_F[-1]/j)>1):
                    
                    # print("appended P",P_F[-1])
                    x1=[mom[index], mom[index+1]]
                    y1=[de_dx[index], de_dx[index+1]]
                    # print("index",index)         
                index+=1
            # print(x1,y1)   
            fx = interpolate.interp1d(x1, y1,kind = 'linear')
            en_loss= fx(P_F[-1])
            # print("en-loss",en_loss)
            en_incident = E_inc[-1]
            # print("EnIncident",en_incident)
            en_atplane = en_incident-(en_loss*step_size/10*rho_fe)
            # print("en_atplane",en_atplane)
            if(en_atplane<105):
                p=303.20151
            else:    
                p=math.sqrt(math.pow((en_atplane),2)-math.pow((mass_mu),2))
            # print("p",p)
            # print("e/p",e/p)
            E_inc.append(en_atplane)           #for filtering
            E_atplane.append(en_atplane)       #for smoothing
            P_F.append(p)
            QP.append(e/p)
            return e/p
        if(i==1):
            P2.append(P_F[i])           
        #########   Step Loop      ######################
       
        for step in range(59):

            if(step==0 or step==58):
                step_size=0.1  #mm
            else:
                step_size=1 #mm
            # print("***********************************")
            # print("step no.",step,"step_size",step_size)
            # print("************************************") 
              
         ###########   Prediction   ##################

           
                        
            # plt.loglog(t,de_dx)
            if(step==0):
                x0_step=x0_lyr
                y0_step=y0_lyr
                tx0_step=tx0_lyr
                ty0_step=ty0_lyr
                qp0_step=qp0_lyr
                
            else:
                # prediction
                x0_step=xp_step[-1]
                y0_step=yp_step[-1]
                tx0_step=txp_step[-1]
                ty0_step=typ_step[-1]
                qp0_step=qpp_step[-1]
            
                      
            
            qp_p=predict_qp_f()
            # print(qp_p)
            x_p=predict_x(x0_step,tx0_step,ty0_step,qp_p,step_size)
            y_p=predict_y(y0_step,tx0_step,ty0_step,qp_p,step_size)
            tx_p=predict_tx(tx0_step,ty0_step,qp_p,step_size)
            ty_p=predict_ty(tx0_step,ty0_step,qp_p,step_size)
            
            # print("Predicted values of step :",step)
            # print("x0:",x0_step,"y0:",y0_step,"tx0:",tx0_step,"ty0:",ty0_step,"qp0:",qp0_step)
            
            # appending the predicted variables for plots
            xp_step.append(x_p)
            yp_step.append(y_p)
            txp_step.append(tx_p)
            typ_step.append(ty_p)
            qpp_step.append(qp_p)
            T=math.sqrt(tx0_step**2+ty0_step**2+1)
            l=step_size
            dl=(step_size)/(1/T)
            # print("T",T,"dl",dl,"l",l)
        
            

            ######################    filtering    ########################




            # PROPAGATOR MATRIX
    
            def propagator():


                tx, ty, qp, dz, x0, y0, del_x, del_y, del_tx, del_ty, del_qp = sp.symbols('tx ty qp dz x0 y0 del_x del_y del_tx del_ty del_qp')

                x_ze = x0 + tx*dz + ((qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz)) + ((k_h*qp*((1+tx**2+ty**2)**0.5))**2)*((tx*(3*ty*2+1)*Sxx(dz)) - (ty*(3*tx**2+1)*Sxy(dz)) - (ty*(3*tx**2+1)*Syx(dz)) + (tx*(3*tx**2+3)*Syy(dz)))
                y_ze = y0 + ty*dz + ((qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))
                tx_ze = tx +((qp*sp.sqrt(tx**2+ty**2+1)))*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))
                ty_ze = ty + ((qp*sp.sqrt(tx**2+ty**2+1)))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))
                # qp_ze = 0
                    
                    
                ####list of constatnts of f(l) the function relating momentum and length travelled for 5GeV (from range-momentum relation of muon in iron)
                fl, l,fl1 ,fl2,c1, c2, c3= sp.symbols('fl l fl1 fl2 c1 c2 c3')
                c1=-3.644e-13
                c2=6.415e-10
                c3=4.437e-07
                c4=0.0001521
                c5=0.02675
                c6=2.227
                    
                fl= -c1*l**5 + c2*l**4 - c3*l**3 + c4*l**2 - c5*l + c6
                fl1=sp.diff(fl,l)
                fl2=sp.diff(fl1,l)
                fl3=sp.diff(fl2,l)
                # print(fl1)
                #  print(fl2)
                #  print(fl3)
                d = {x0: sp.symbols("del_x"), y0: sp.symbols("del_y"), tx: sp.symbols("del_tx"), ty: sp.symbols("del_ty"), qp: sp.symbols("del_qp")}
                d_qp = (1+((fl2/fl1*dl)+(1/2*fl3/fl1)*dl**2))*d[qp] + ((fl1+fl2*dl)*fl*T*dl*(-Bx*d[x0]+Bz*d[y0])) + (fl1+fl2*dl)*step_size*(tx0/T*d[tx]+ty0/T*d[ty])
                # print(d_qp)
                    
                deleqn=[]
                    
                param = (x_ze, y_ze, tx_ze, ty_ze)
                    
                for j in(param):
                    PD=0
                    for i in (x0,y0,tx,ty,qp):
                        PD += sp.diff(j, i)*d[i]
                        # F.fill(sp.diff(PD, k))
                        deleqn.append(PD)
                        # print(PD)
                        # print("")
                    # print(deleqn[2])
                deleqn.append(d_qp)
                # print(deleqn[4])
                    
                # deno = (del_x, del_y, del_tx, del_ty, del_qp)
                
                
                #part for extracting the equations and substituting the global variables in the eqn
                    
                    
            
                # f00 = sp.diff(deleqn[0],deno[0]) ; print("f00\n",f00)    
                # f01 = sp.diff(deleqn[0],deno[1]) ; print("f01\n",f01)
                # f02 = sp.diff(deleqn[0],deno[2]) ; print("f02\n",f02)
                # f03 = sp.diff(deleqn[0],deno[3]) ; print("f03\n",f03)
                # f04 = sp.diff(deleqn[0],deno[4]) ; print("f04\n",f04)
                        
                # f10 = sp.diff(deleqn[1],deno[0]) ; print("f10\n",f10)
                # f11 = sp.diff(deleqn[1],deno[1]) ; print("f11\n",f11)
                # f12 = sp.diff(deleqn[1],deno[2]) ; print("f12\n",f12)
                # f13 = sp.diff(deleqn[1],deno[3]) ; print("f13\n",f13)
                # f14 = sp.diff(deleqn[1],deno[4]) ; print("f14\n",f14)
                
                # f20 = sp.diff(deleqn[2],deno[0]) ; print("f20\n",f20)
                # f21 = sp.diff(deleqn[2],deno[1])  ; print("f21\n",f21)
                # f22 = sp.diff(deleqn[2],deno[2])  ; print("f22\n",f22)
                # f23 = sp.diff(deleqn[2],deno[3])  ; print("f23\n",f23)
                # f24 = sp.diff(deleqn[2],deno[4])  ; print("f24\n",f24)
                    
                # f30 = sp.diff(deleqn[3],deno[0])  ; print("f30\n",f30)
                # f31 = sp.diff(deleqn[3],deno[1])  ; print("f31\n",f31)
                # f32 = sp.diff(deleqn[3],deno[2])  ; print("f32\n",f32)
                # f33 = sp.diff(deleqn[3],deno[3])  ; print("f33\n",f33)
                # f34 = sp.diff(deleqn[3],deno[4])  ; print("f34\n",f34)

                # f40 = sp.diff(deleqn[4],deno[0])  ; print("f40\n",f40)
                # f41 = sp.diff(deleqn[4],deno[1])  ; print("f41\n",f41)
                # f42 = sp.diff(deleqn[4],deno[2])  ; print("f42\n",f42)
                # f43 = sp.diff(deleqn[4],deno[3])  ; print("f43\n",f43)
                # f44 = sp.diff(deleqn[4],deno[4])  ; print("f44\n",f44)


        
                    
                return  0
            
            def gf00():
                return 1
            def gf01():
                return 0
            def gf02():
                return 674055330.75*step_size**3*qp_p**2*tx0_step**2*(6*ty0_step + 1) + 337027665.375*step_size**3*qp_p**2*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1 + 22484.25*step_size**2*qp_p*tx0_step**2*ty0_step/(math.sqrt(tx0_step**2 + ty0_step**2 + 1))+ 0.75*step_size**2*qp_p*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + step_size
            def gf03():
                return 674055330.75*step_size**3*qp_p**2*tx0_step*ty0_step*(6*ty0_step + 1) + 2022165992.25*step_size**3*qp_p**2*tx0_step*(tx0_step**2 + ty0_step**2 + 1)**1 + 22484.25*step_size**2*qp_p*tx0_step*ty0_step**2/(math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + 0.75*step_size**2*qp_p*tx0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
            def gf04():  
                return 674055330.75*step_size**3*qp_p*tx0_step*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1 + 22484.25*step_size**2*tx0_step*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
            
            def gf10():
                return 0
            def gf11():
                return 1
            def gf12():
                return 674055330.75*step_size**3*qp_p**2*tx0_step*ty0_step*(3*ty0_step**2 + 3) + 22484.25*step_size**2*qp_p*tx0_step*(ty0_step**2 + 1)/(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
            def gf13():
                return 674055330.75*step_size**3*qp_p**2*ty0_step**2*(3*ty0_step**2 + 3) + 2022165992.25*step_size**3*qp_p**2*ty0_step**2*(tx0_step**2 + ty0_step**2 + 1)**1 + 337027665.375*step_size**3*qp_p**2*(3*ty0_step**2 + 3)*(tx0_step**2 + ty0_step**2 + 1)**1 + 22484.25*step_size**2*qp_p*ty0_step*(ty0_step**2 + 1)/(math.sqrt(tx0_step**2 + ty0_step**2 + 1) )+ 44968.5*step_size**2*qp_p*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + step_size
            def gf14():
                return 674055330.75*step_size**3*qp_p*ty0_step*(3*ty0_step**2 + 3)*(tx0_step**2 + ty0_step**2 + 1)**1 + 22484.25*step_size**2*(ty0_step**2 + 1)*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
              
            def gf20():
                return 0
            def gf21():
                return 0
            def gf22():
                return 2022165992.25*step_size**2*qp_p**2*tx0_step**2*(6*ty0_step + 1) + 1011082996.125*step_size**2*qp_p**2*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1 + 44968.5*step_size*qp_p*tx0_step**2*ty0_step/math.sqrt(tx0_step**2 + ty0_step**2 + 1) + 44968.5*step_size*qp_p*ty0_step*math.sqrt(tx0_step**2 + ty0_step**2 + 1) + 1
            def gf23():
                return 2022165992.25*step_size**2*qp_p**2*tx0_step*ty0_step*(6*ty0_step + 1) + 6066497976.75*step_size**2*qp_p**2*tx0_step*(tx0_step**2 + ty0_step**2 + 1)**1+ 44968.5*step_size*qp_p*tx0_step*ty0_step**2/math.sqrt(tx0_step**2 + ty0_step**2 + 1) +44968.5*step_size*qp_p*tx0_step*math.sqrt(tx0_step**2 + ty0_step**2 + 1)
            def gf24():
                return 2022165992.25*step_size**2*qp_p*tx0_step*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1 + 44968.5*step_size*tx0_step*ty0_step*math.sqrt(tx0_step**2 + ty0_step**2 + 1)
            
            def gf30():
                return 0
            def gf31():
                return 0
            def gf32():
#                return 2022165992.25*step_size**2*qp_p**2*tx0_step*ty0_step*(6*ty0_step + 3) + 44968.5*step_size*qp_p*tx0_step*(ty0_step**2 + 1)*(1/math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + 1.5*step_size*qp_p*tx0_step*(ty0_step**2 + 1)/math.sqrt(tx0_step**2 + ty0_step**2 + 1)
                return  2022165992.25*step_size**2*qp_p**2*tx0_step*ty0_step*(6*ty0_step + 3) + 44968.5*step_size*qp_p*tx0_step*(ty0_step**2 + 1)*(1/math.sqrt(tx0_step**2 + ty0_step**2 + 1)) 
            def gf33():
#                return 2022165992.25*step_size**2*qp_p**2*ty0_step**2*(6*ty0_step + 3) + 6066497976.75*step_size**2*qp_p**2*ty0_step*(tx0_step**2 + ty0_step**2 + 1)**1 + 1011082996.125*step_size**2*qp_p**2*(6*ty0_step + 3)*(tx0_step**2 + ty0_step**2 + 1)**1 + 44968.5*step_size*qp_p*ty0_step*(ty0_step**2 + 1)*(1/math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + 1.5*step_size*qp_p*ty0_step*(ty0_step**2 + 1)/math.sqrt(tx0_step**2 + ty0_step**2 + 1) + 3.0*step_size*qp_p*ty0_step*math.sqrt(tx0_step**2 + ty0_step**2 + 1) + 89937.0*step_size*qp_p*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + 1
                return 2022165992.25*step_size**2*qp_p**2*ty0_step**2*(6*ty0_step + 3)  +  6066497976.75*step_size**2*qp_p**2*ty0_step*(tx0_step**2 + ty0_step**2 + 1)**1 +1011082996.125*step_size**2*qp_p**2*(6*ty0_step + 3)*(tx0_step**2 + ty0_step**2 + 1)**1+  44968.5*step_size*qp_p*ty0_step*(ty0_step**2 + 1)*(1/math.sqrt(tx0_step**2 + ty0_step**2 + 1))  + 89937.0*step_size*qp_p*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1)) + 1
            def gf34():
#                return 2022165992.25*step_size**2*qp_p*ty0_step*(6*ty0_step + 3)*(tx0_step**2 + ty0_step**2 + 1)**1 + 1.5*step_size*(ty0_step**2 + 1)*math.sqrt(tx0_step**2 + ty0_step**2 + 1) + 44968.5*step_size*(ty0_step**2 + 1)*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
                return 2022165992.25*step_size**2*qp_p*ty0_step*(6*ty0_step + 3)*(tx0_step**2 + ty0_step**2 + 1)**1 + 44968.5*step_size*(ty0_step**2 + 1)*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))
            
            def gf40():
                return 144.011361674193*(1.822e-12*l**4 + 3.26567560074942e-9*l**3 - 5.92062846519066e-7*l**2 + 4.86187178491891e-5*l + 0.00245435205103926)*(3.644e-13*l**5 + 6.415e-10*l**4 - 4.437e-7*l**3 + 0.0001521*l**2 - 0.02675*l + 2.227)
            def gf41():
                return 0
            def gf42():
                return -1.24489251563824e-12*l**4 - 2.23129259817529e-9*l**3 + 4.04530519439659e-7*l**2 - 3.321903291459e-5*l - 0.00167695087765101
            def gf43():
                return 9.29489610327295e-13*l**4 + 1.66597779450929e-9*l**3 - 3.02039662184564e-7*l**2 + 2.48027404545859e-5*l + 0.00125208272860948
            def gf44():
                return  9216.72714714835*(1.0932e-11*l**2 + 7.698e-9*l - 1.3311e-6)/(1.822e-12*l**4 + 2.566e-9*l**3 - 1.3311e-6*l**2 + 0.0003042*l - 0.02675) + 96.0037871500304*(7.288e-12*l**3 + 7.698e-9*l**2 - 2.6622e-6*l + 0.0003042)/(1.822e-12*l**4 + 2.566e-9*l**3 - 1.3311e-6*l**2 + 0.0003042*l - 0.02675) + 1

            F=[[gf00(),gf01(),gf02(),gf03(),gf04()],[gf10(),gf11(),gf12(),gf13(),gf14()],[gf20(),gf21(),gf22(),gf23(),gf24()],[gf30(),gf31(),gf32(),gf33(),gf34()],[gf40(),gf41(),gf42(),gf43(),gf44()]]
            # print(" ")
            # print("matrix F",F)
            # print("")
            F_matrix.append(F)
            # print("  ")
            # print("matrix F",F_matrix)
            # print("Deter F",np.linalg.det(F))
                        
        # TERMS IN NOISE MATRIX
            
            # terms due to multiple scattering 
            D = -1      # forward direction
            lQ = step_size*T*D
            ls = rl_fe*((Z_fe+1)/Z_fe)*(289*np.power(Z_fe,1/3))/(159*np.power(Z_fe,1/2))
            CMS = ((0.015)**2/((beta)**2*((P_F[i])**2)))*lQ/ls

            # terms due to Energy loss straggling

            Tmax = 2*mass_e*(beta)**2*(gamma)**2/(1+(2*gamma*mass_e/mass_mu)+(mass_e/mass_mu)**2)
            si = 0.1534*e**2*Z_fe/(beta**2*A_fe)*rho_fe*d_fe
            sigma_sq_E = si*Tmax*(1-(beta**2/2))

            # x00, y00, x, y, tx, ty, tx1, ty1, q, P, dz = sp.symbols('x00 y00 x y tx ty tx1 ty1 q P dz ')
            
            # x = x00 + tx*dz + (k_h*q/P*((1+tx**2+ty**2)**0.5)) *(tx*ty*Sx(dz)- (1+tx**2)*Sy(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(tx*(3*ty*2+1)*Sxx(dz) - ty*(3*tx**2+1)*Sxy(dz) - ty*(3*tx**2+1)*Syx(dz) + tx*(3*tx**2+3)*Syy(dz))
            # y=y00 + ty*dz + (k_h*q/P*((1+tx**2+ty**2)**0.5))*((1+ty**2)*Sx(dz)- tx*ty*Sy(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(ty*(3*ty**2+3)*Sxx(dz) - tx*(3*ty**2+1)*Sxy(dz) - tx*(3*ty**2+1)*Syx(dz) + ty*(3*tx**2+1)*Syy(dz))
            # tx1 = tx + (k_h*q/P*((1+tx**2+ty**2)**0.5))*(tx*ty*Rx(dz)- (1+tx**2)*Ry(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(tx*(3*ty*2+1)*Rxx(dz) - ty*(3*tx**2+1)*Rxy(dz) - ty*(3*tx**2+1)*Ryx(dz) + tx*(3*tx**2+3)*Ryy(dz))
            # ty1=ty + (k_h*q/P*((1+tx**2+ty**2)**0.5))*((1+ty**2)*Rx(dz)- tx*ty*Ry(dz))+ (k_h*q/P*((1+tx**2+ty**2)**0.5))**2*(ty*(3*ty*2+3)*Rxx(dz) -tx*(3*tx**2+1)*Rxy(dz) - tx*(3*ty**2+1)*Ryx(dz) + ty*(3*tx**2+1)*Ryy(dz))
        
            dx_dp = -0.2248425*(step_size)**2*e*tx0_step*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))/P_F[i]**2 - 0.067405533075*(step_size)**3*e**2*tx0_step*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1/P_F[i]**3
            dy_dp = -0.2248425*step_size**2*e*(ty0_step**2 + 1)*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))/P_F[i]**2 - 0.067405533075*step_size**3*e**2*ty0_step*(3*ty0_step**2 + 3)*(tx0_step**2 + ty0_step**2 + 1)**1/P_F[i]**3
            dtx1_dp =  -0.449685*step_size*e*tx0_step*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))/P_F[i]**2 - 0.202216599225*step_size**2*e**2*tx0_step*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1/P_F[i]**3
            dty1_dp =  -0.449685*step_size*e*tx0_step*ty0_step*(math.sqrt(tx0_step**2 + ty0_step**2 + 1))/P_F[i]**2 - 0.202216599225*step_size**2*e**2*tx0_step*(6*ty0_step + 1)*(tx0_step**2 + ty0_step**2 + 1)**1/P_F[i]**3

            
            c_xqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dx_dp)*sigma_sq_E
            c_yqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dy_dp)*sigma_sq_E
            c_txqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dtx1_dp)*sigma_sq_E
            c_tyqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*(dty1_dp)*sigma_sq_E
            c_qpqp= (-e/P_F[i]**2)*((E_inc[i]/P_F[i])**2)*sigma_sq_E


            def Q00():
                return (1+tx0_step**2)*T**2*CMS*lQ**3/3
            def Q01():
                return  (tx0_step*ty0_step)*T**2*CMS*lQ**3/3
            def Q02():
                return  (1+tx0_step**2)*T**2*CMS*lQ**2/2*D
            def Q03():
                return  (tx0_step*ty0_step)*T**2*CMS*lQ**2/2*D
            def Q04():
                return  c_xqp
            def Q10():
                return  (tx0_step*ty0_step)*T**2*CMS*lQ**3/3
            def Q11():
                return  (1+ty0_step**2)*T**2*CMS*lQ**3/3  
            def Q12():
                return  (tx0_step*ty0_step)*T**2*CMS*lQ**2/2*D
            def Q13():
                return  (1+ty0_step**2)*T**2*CMS*lQ**2/2*D
            def Q14():
                return  c_yqp

            def Q20():
                return  (1+tx0_step**2)*T**2*CMS*lQ**2/2*D
            def Q21():
                return  (tx0_step*ty0_step)*T**2*CMS*lQ**2/2*D
            def Q22():
                return   (1+tx0_step**2)*T**2*CMS*lQ
            def Q23():
                return   (tx0_step*ty0_step)*T**2*CMS*lQ
            def Q24():
                return    c_txqp

            def Q30():
                return   (tx0_step*ty0_step)*T**2*CMS*lQ**2/2*D
            def Q31():
                return   (1+ty0_step**2)*T**2*CMS*lQ**2/2*D
            def Q32():
                return   (tx0_step*ty0_step)*T**2*CMS*lQ
            def Q33():
                return   (tx0_step*ty0_step)*T**2*CMS*lQ
            def Q34():
                return    c_tyqp  

            def Q40():
                return  c_xqp 
            def Q41():
                return  c_yqp
            def Q42():
                return  c_txqp
            def Q43():
                return  c_tyqp 
            def Q44():
                return  c_qpqp        


            Qk=[[Q00(),Q01(),Q02(),Q03(),Q04()],
            [Q10(),Q11(),Q12(),Q13(),Q14()],
            [Q20(),Q21(),Q22(),Q23(),Q24()],
            [Q30(),Q31(),Q32(),Q33(),Q34()],
            [Q40(),Q41(),Q42(),Q43(),Q44()]]
            
            Qk_matrix.append(Qk)
            # print("  ")
            # print("QK",Qk) 
            
            x_k0 = np.transpose([[x0_step,y0_step,tx0_step,ty0_step,qp0_step]])
            # print("SV x_k0:",x_k0)
            x_k1 = np.transpose([[x_p,y_p,tx_p,ty_p,qp_p]])       
            # print("SV x_k1:",x_k1)
            x_k1_step.append(x_k1)
            
            if(i==0):   
                c_k0=np.zeros((5,5) ,float)
                np.fill_diagonal(c_k0,10**6)
                
            else:
                c_k0=Ck_matrix[-1]
                # c_k0 = (x_k1-x_k0)*(np.transpose(x_k1-x_k0))
            # print("EC1",c_k0)    
            
            # print("F_matrix",F)
            # print(FN)
            # print(F_matrix[-1])
            # FN=FN*np.array(F_matrix[-1])
            # print(FN)
            # if(step==1):
            #     break
                    
            if(step==0 or step==58):   
                c_k1 = F*np.array(c_k0)*(np.transpose(F)) 

            else:
                # c_k1 = F*np.array(c_k0)*(np.transpose(F)) + FN*np.array(Qk)*np.transpose(FN)
                c_k1 = F*np.array(c_k0)*(np.transpose(F)) + F*np.array(Qk)*np.transpose(F)
            
            c_k1_step.append(c_k1)
            # print("EC2",c_k1)
        # print(F_matrix) 
        # print("")   
        # print(c_k1_step)
        P_F_lyr.append(P_F[-1])
        # print("p:",P_F[-1])
        
            
        H=np.matlib.eye(n = 2, M = 5, k = 0, dtype = float)
        V=[[28*28/12,0],[0,28*28/12]]
        # m_k = [[snx[i]],[sny[i]]]
        m_k = [[eid_a[(i+1),6]],[eid_a[(i+1),4]]]
        kg2=(H*np.array(c_k1)*np.transpose(H)+V).astype(np.float64)    # kg2 was trated as obj instead of float and hence the conversion
        k_gain= np.array(c_k1)*np.transpose(H)*np.linalg.inv(kg2) 
        # print("k_gain",k_gain)
        # filtered state vector
        x_k = x_k1_step[-1] + k_gain*(m_k-H*x_k1_step[-1])
        # print("State vector of",i+1,"plane:\n",x_k)
        #  filter_state_vector=x_k
        
        xf_fwd=x_k[0,0]
        yf_fwd=x_k[1,0]
        txf_fwd=x_k[2,0]
        tyf_fwd=x_k[3,0]
        qpf_fwd=x_k[4,0]


        
        I= np.matlib.eye(n = 5, M = 5, k = 0, dtype = float)
        Ck = (I-k_gain*H)*c_k1*(np.transpose(I-k_gain*H))+k_gain*V*(np.transpose(k_gain))
        # print((I-k_gain*H)*c_k1*(np.transpose(I-k_gain*H)))
        # print("error Covariance:\n",Ck) 
        Ck_matrix.append(Ck)
     #     # print("\neid:",eid_a[i,0])
     #     # print("\nq:",x_k[4]*P_F[i])
     #     # print("\nP from state r:",x_k[4]*P_F[i]/x_k[4])
    # print("Event:",n)
#    print("P",P_F_lyr[-1])
    diff_p=((eid_a[:,19]**2+eid_a[:,18]**2+eid_a[:,20]**2)**0.5)-np.array(P_F_lyr)/1000
#    diff_eve.append(diff_p)
#    print("diff",diff_p)    
    mu_event.append(np.mean(np.array(diff_p)))  ##for single  event
    sigma_event.append(np.std(np.array(diff_p)))##for single event
mu=np.mean(np.array(mu_event)) ## for all events pertaining to a particular energy regime
sigma=np.std(np.array(sigma_event)) ## for all events pertaining to a particular energy regime
print("mean:",mu)
print("sigma:",sigma)
s = np.random.normal(mu, sigma, 10000)
count, bins, ignored = plt.hist(s, 300, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
# x=((eid_a[:,19]**2+eid_a[:,18]**2+eid_a[:,20]**2)**0.5)
# plt.plot(x,y)
# plt.hist(y)
plt.xlabel("true P- Kalman P")
plt.ylabel("Frequency")
plt.show()

# # plt.plot(diff_eve,mu)
# # plt.xlabel("TrueP-KalmanP")
# # plt.ylabel("Mu")
# # plt.show()
# # plt.plot(diff_eve,sigma)
# # plt.xlabel("TrueP-KalmanP")
# # plt.ylabel("Sigma")
# # plt.show()

    
      
 