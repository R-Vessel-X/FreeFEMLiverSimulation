# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:57:58 2022

@author: M.Chetoui

The essential function of this library is elementary_parametrization3D. this function
works on a given elementary volume, defined with an origin and 3 directional steps, and with a segment
of the virtual trees defined by its radius and the coordinates of its two limit points.
This function verify if the segment of a part of it is included in the elementary volume using geometrical
 coordinates-based operations.
Then, the function computes some geometrical quantities of the included part that is necessary to define the permeability
and the coupling coefficient.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def elementary_parametrization3D(O, step, p1, p2, D):
    
    #Initialize the coordinates of the two limits of the segment
    N=np.array([[100.,100.,100.],[100.,100.,100.]])
    #Define the elementary volume space
    Lim=np.array([[O[0]-step[0],O[0]+step[0]],[O[1]-step[1],O[1]+step[1]],[O[2]-step[2],O[2]+step[2]]])

    #Tests to verify if one or both of the segment terminal nodes are inside the domain
    Testx1=((Lim[0,0])<=p1[0]) and ((Lim[0,1])>=p1[0])
    Testx2=((Lim[0,0])<=p2[0]) and ((Lim[0,1])>=p2[0])
    Testy1=((Lim[1,0])<=p1[1]) and ((Lim[1,1])>=p1[1])
    Testy2=((Lim[1,0])<=p2[1]) and ((Lim[1,1])>=p2[1])
    Testz1=((Lim[2,0])<=p1[2]) and ((Lim[2,1])>=p1[2])
    Testz2=((Lim[2,0])<=p2[2]) and ((Lim[2,1])>=p2[2])

    Comp=0
    #First case: the first or the second limit of the segment is in the domain or both of them
    if ((Testx1 and Testy1 and Testz1)):
        #print('T1, comp= ', Comp)
        N[Comp,:]=p1
        Comp+=1
        
        
    
    if ((Testx2 and Testy2 and Testz2)):
        #print('T2, comp= ', Comp)
        N[Comp,:]=p2
        Comp+=1
        
    
    #Second case: search the intersection between the segment and the elementary volume  
    direct=np.array([p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]) #Segment director vector
    for i in range(3):
            
        if Comp==2:
            break
            
        elif (direct[i]==0): #Director vector normal to an axis
            if p1[i]>=Lim[i,0] and p1[i]<=Lim[i,1]:
                
                for j in range(3):
                    k=-1
                    while ((k<0 or k==i or k==j) and i!=j):
                        k+=1
                    if (j>i and direct[j]==0): #Director vector normal to a second axis
                        if p1[j]>=Lim[j,0] and p1[j]<=Lim[j,1]:
                            for m in range(2):
                                if (Lim[k,m]<=np.max([p1[k],p2[k]]) and Lim[k,m]>=np.min([p1[k],p2[k]])):
                                    if (Comp==0) or ((Comp>0) and (N[Comp-1,k]!=Lim[k,m])):
                                        #print('T3, comp= ', Comp)
                                        N[Comp,i]=p1[i]
                                        N[Comp,j]=p1[j]
                                        N[Comp,k]=Lim[k,m]
                                        Comp+=1
                                        
                                    
                    elif (j!=i): #Director vector normal to only one axis
                        for q in [j,k]:
                            for m in range(2):
                                if (Lim[q,m]<=np.max([p1[q],p2[q]]) and Lim[q,m]>=np.min([p1[q],p2[q]])):
                                    if (Comp==0) or ((Comp==1) and (N[Comp-1,q]!=Lim[q,m])):
                                        if q==j:
                                            A=(direct[k]/direct[q])*(Lim[q,m]-p1[q])+p1[k]
                                            if (A>=Lim[k,0]) and (A<=Lim[k,1]):
                                                #print('T4, comp= ', Comp)
                                                N[Comp,i]=p1[i]
                                                N[Comp,j]=Lim[q,m]
                                                N[Comp,k]=A
                                                Comp+=1
                                                
                                            
                                        else:
                                            A=(direct[j]/direct[q])*(Lim[q,m]-p1[q])+p1[j]
                                            if (A>=Lim[j,0]) and (A<=Lim[j,1]):
                                                #print('T5, comp= ', Comp)
                                                N[Comp,i]=p1[i]
                                                N[Comp,k]=Lim[q,m]
                                                N[Comp,j]=A
                                                Comp+=1
                                                   
                                    
        else: #Director vector has 3 components
            j=-1
            k=-1
            for m in range(2):
                if (Lim[i,m]<=np.max([p1[i],p2[i]]) and Lim[i,m]>=np.min([p1[i],p2[i]])):
                    if (Comp==0) or ((Comp==1) and (N[Comp-1,i]!=Lim[i,m-1])):
                        while (j<0 or j==i):
                            j+=1
                        while ((k<0 or k==i or k==j) and i!=j):
                            k+=1
                                
                        A=(direct[j]/direct[i])*(Lim[i,m]-p1[i])+p1[j]
                        B=(direct[k]/direct[i])*(Lim[i,m]-p1[i])+p1[k]
                            
                        if (((B>=Lim[k,0]) and (B<=Lim[k,1]))  and ((A>=Lim[j,0]) and (A<=Lim[j,1]))):
                            #print('T6, comp= ', Comp)
                            N[Comp,i]=Lim[i,m]
                            N[Comp,j]=A
                            N[Comp,k]=B
                            Comp+=1
                            
                                
                
    if Comp==2:
        L= np.sqrt(np.square(N[1,0]-N[0,0])+np.square(N[1,1]-N[0,1])+np.square(N[1,2]-N[0,2]))
        
    else:
        L=0
    
    V=L*np.pi*(D**2)/4
    
    return V*1e-9,L*1e-3,D*1e-3,np.abs(N[0,0]-N[1,0])*1e-3,np.abs(N[0,1]-N[1,1])*1e-3,np.abs(N[0,2]-N[1,2])*1e-3;


def TwoDfrom3D(NAxis,XXX,YYY,Z,XYZGRID,DATA, Xaxis, Yaxis, Figtitle, Figname):
    
    XX, YY = np.meshgrid(YYY, XXX)
    XX1=np.reshape(XX, (np.size(XXX)*np.size(YYY)))
    YY1=np.reshape(YY, (np.size(XXX)*np.size(YYY)))
    Newtab=np.ones((np.size(XXX[:])*np.size(YYY[:]),4),dtype=float)
    

    if NAxis==0:
        c1=1
        c2=2
    elif NAxis==1:
        c1=2
        c2=0
    elif NAxis==2:
        c1=0
        c2=1
        
    Newtab[:,c1]=XX1
    Newtab[:,c2]=YY1
    Newtab[:,NAxis]=Newtab[:,NAxis]*Z
    
    linInter= LinearNDInterpolator(XYZGRID, DATA)
    
    
    for j in range(np.size(Newtab[:,1])):
        Newtab[j,3]=linInter(Newtab[j,0:3])
                            
            
        
    RESL=np.reshape(Newtab[:,3],   (np.size(XXX),np.size(YYY)))
    fig, ax=plt.subplots()
    fig.set_dpi(200)
    fig.set_size_inches(8,6, forward=True)

    plt.contourf(XX,YY,RESL)
    plt.colorbar()

    #plt.legend(loc='upper right');
    #plt.grid(axis='both')
    ax.set_xlabel(Xaxis)
    ax.set_ylabel(Yaxis)
    plt.title(Figtitle)

    plt.savefig(Figname)
    
    return(Newtab,c1,c2)