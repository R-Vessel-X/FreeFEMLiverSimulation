# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:56:13 2023

@author: M.Chetoui

This algorithm solve the System of nn unknows with nn= total number of 1D nodes of the PV and the HV tree
multiplied by 2 (velocity + pressure). it helps, using Darcy system obtained from 3D simulation,
Bernoulli equations and continuity equations to find the pressure and the velocity in the large vessels
and therefore to compare the numerical results with clinical ordinary metrics.
The algorithm uses the Darcy matrix obtained by the FE simulation and by the properties
of the 1D trees of HV and PV.
"""

import numpy as np
from scipy.optimize import fsolve

################### Physical properties 

nu=3.6*1e-2  #viscosity in Pa.s
rho=1060.0  #Density in kg/m3

################### 1D trees properties reading


AVP=np.loadtxt('E1D/PV_Arb.txt')   #PV 1D tree
AVH=np.loadtxt('E1D/HV_Arb.txt')   #HV 1D tree


CVP=np.loadtxt('E1D/PV_conn.txt', dtype=int)  #Connectivity of PV
CVH=np.loadtxt('E1D/HV_conn.txt', dtype=int)  #Connectivity of HV


################### Darcy matrix reading


factor=10.0 ####### correction factor for Darcy system
Darcy=np.loadtxt('FE_model/Darcy_example.txt')*factor


################## Initialization 

nvp=np.size(AVP[:,0])                  #velocity unkowns number PV
nvh=np.size(AVH[:,0])                  #velocity unkowns number HV

nn=2*(nvh+nvp);                              #Total unknowns

K1=np.zeros((nn,nn) , dtype=float)         #First matrix of the system
K2=np.zeros((nn,nn) , dtype=float)         #Second matrix of the system

Cl=np.zeros((nn,), dtype=float);          #Boundary conditions matrix


####################### matrices filling  ################################

j=0            #Equations counter

##### Continuity equation

# V0*D0²= V1*D1² + V2*D2²     ====>  -V0*D0² + V1*D1² + V2*D2² = 0



for i in range(np.size(CVP[:,0])):
    
    K1[j,CVP[i,0]]=-AVP[CVP[i,0],3]**2
    K1[j,CVP[i,1]]=AVP[CVP[i,1],3]**2
    K1[j,CVP[i,2]]=AVP[CVP[i,2],3]**2
    
    j=j+1
    
for i in range(np.size(CVH[:,0])):
    
    K1[j,CVH[i,0]+nvp]=-AVH[CVH[i,0],3]**2
    K1[j,CVH[i,1]+nvp]=AVH[CVH[i,1],3]**2
    K1[j,CVH[i,2]+nvp]=AVH[CVH[i,2],3]**2
    
    j=j+1
    
############ Bernoulli equations

# p0 + (rho/2)*V0²= p1 + (rho/2)*V1² + 32*rho*nu*(L1/D1²)*V1   ====>  p0 + (rho/2)*V0²- p1 - (rho/2)*V1² - 32*rho*nu*(L1/D1²)*V1 = 0

for i in range(np.size(CVP[:,0])):
    
    l1=np.sqrt((AVP[CVP[i,0],0]-AVP[CVP[i,1],0])**2+(AVP[CVP[i,0],1]-AVP[CVP[i,1],1])**2+(AVP[CVP[i,0],2]-AVP[CVP[i,1],2])**2)
    l2=np.sqrt((AVP[CVP[i,0],0]-AVP[CVP[i,2],0])**2+(AVP[CVP[i,0],1]-AVP[CVP[i,2],1])**2+(AVP[CVP[i,0],2]-AVP[CVP[i,2],2])**2)
    
    K2[j,CVP[i,0]]=(rho/2)*1e-6;
    K2[j,CVP[i,1]]=(-rho/2)*1e-6;
    K1[j,CVP[i,1]]=32*nu*l1/(2*AVP[CVP[i,1],3])**2;
    K1[j,CVP[i,0]+nvp+nvh]=1;
    K1[j,CVP[i,1]+nvp+nvh]=-1;
    
    j=j+1
    
    K2[j,CVP[i,0]]=(rho/2)*1e-6;
    K2[j,CVP[i,2]]=(-rho/2)*1e-6;
    K1[j,CVP[i,2]]=32*nu*l2/(2*AVP[CVP[i,2],3])**2;
    K1[j,CVP[i,0]+nvp+nvh]=1;
    K1[j,CVP[i,2]+nvp+nvh]=-1;
    
    j=j+1
    
for i in range(np.size(CVH[:,0])):
    
    l1=np.sqrt((AVH[CVH[i,0],0]-AVH[CVH[i,1],0])**2+(AVH[CVH[i,0],1]-AVH[CVH[i,1],1])**2+(AVH[CVH[i,0],2]-AVH[CVH[i,1],2])**2)
    l2=np.sqrt((AVH[CVH[i,0],0]-AVH[CVH[i,2],0])**2+(AVH[CVH[i,0],1]-AVH[CVH[i,2],1])**2+(AVH[CVH[i,0],2]-AVH[CVH[i,2],2])**2)
    
    K2[j,CVH[i,0]+nvp]=(rho/2)*1e-6;
    K2[j,CVH[i,1]+nvp]=-(rho/2)*1e-6;
    K1[j,CVH[i,1]+nvp]=32*nu*l1/(2*AVH[CVH[i,1],3])**2;
    K1[j,CVH[i,0]+2*nvp+nvh]=1;
    K1[j,CVH[i,1]+2*nvp+nvh]=-1;
    
    j=j+1
    
    K2[j,CVH[i,0]+np.size(AVP[:,0])]=(rho/2)*1e-6;
    K2[j,CVH[i,2]+np.size(AVP[:,0])]=-(rho/2)*1e-6;
    K1[j,CVH[i,2]+np.size(AVP[:,0])]=32*nu*l2/(2*AVH[CVH[i,2],3])**2;
    K1[j,CVH[i,0]+2*nvp+nvh]=1;
    K1[j,CVH[i,2]+2*nvp+nvh]=-1;
    
    j=j+1
    

######### Darcy Equation (Be careful when put the right column number)

# V2= a0*pe2 + a1*pe3 + a2*pe4 + a3*ps2 + a4*ps3 + a5*ps4    ====>  -V2+ a0*pe2 + a1*pe3 + a2*pe4 + a3*ps2 + a4*ps3 + a5*ps4=0
# Ve: 24:48 (25)    ;    Vs: 68:88 (21)    ;    Pe:  113:137  (25)   ;       Ps: 157:177 (21)

for i in range(25):
    K1[j,i+24]=-1;
    for k in range(25):
        K1[j,k+113]=Darcy[i,k];
    for k in range(21):
        K1[j,k+157]=Darcy[i,k+25];

    j+=1

for i in range(21):
    K1[j,i+68]=-1;
    for k in range(25):
        K1[j,k+113]=Darcy[i+25,k];
    for k in range(21):
        K1[j,k+157]=Darcy[i+25,k+25];
    j+=1


################# Boundary conditions
K1[j,0]=1;     #Inlet velocity PV defined
K1[j+1,138]=1; #Outlet pressure HV defined
K1[j+2,146]=1;  #Second Outlet pressure HV defined

Cl[j]=-200;     #Inlet velocity PV value (negative)
Cl[j+1]=665;      #First Outlet pressure HV value
Cl[j+2]=665;      #Second Outlet pressure HV value

########## Resolution

minC=np.zeros((nn,))
maxC=np.zeros((nn,))
x0=np.zeros((nn,))


########################### adding some constraints
kk=0
while kk<nvp:
    minC[kk]=-250
    maxC[kk]=0
    x0[kk]=-200
    kk+=1
    
while kk<nvp+nvh:
    minC[kk]=10
    maxC[kk]=200
    x0[kk]=100
    kk+=1
    
while kk<2*nvp+nvh:
    minC[kk]=900
    maxC[kk]=1700
    x0[kk]=1330
    kk+=1
    
while kk<nn:
    minC[kk]=665
    maxC[kk]=1000
    x0[kk]=665
    kk+=1


#################### system definition
def Sys(x):
    return np.dot(K1,x)+np.dot(K2,(np.multiply(x,x)))-Cl


#x1=Sys(x0)




############# Resolution
x1, infodict, ier, mesg = fsolve(Sys, x0, full_output=True)
residu=infodict['fvec']

Ve=x1[0:nvp];   ######### PV 1D velocities
Vs=x1[nvp:nvh+nvp];       ######### HV 1D velocities
Pe=x1[nvh+nvp:nvh+2*nvp];   ######### PV 1D pressures
Ps=x1[nvh+2*nvp:nn];       ######### HV 1D pressures

np.savetxt('Final_1D_results/Vin.txt',Ve,fmt='%1.5f') 
np.savetxt('Final_1D_results/Vout.txt',Vs,fmt='%1.5f') 
np.savetxt('Final_1D_results/Pin.txt',Pe,fmt='%1.5f') 
np.savetxt('Final_1D_results/Pout.txt',Ps,fmt='%1.5f') 
