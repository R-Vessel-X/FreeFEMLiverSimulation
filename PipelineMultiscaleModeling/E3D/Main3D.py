# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:21:50 2022

@author: M.Chetoui

This Algorithm permit the user to define the multicompartment properties:
the permeability distributions of each compartment and the coupling coefficients
from OpenCCO .xml file describing the virtual PV and the HV trees.
 The algorithm calls 3 functions:EI and EF from the library Export_DATA and EP3
from the library Fragment_volume. EI permit the user to extract the connectivities
and the CCO provided information. EF, to organize the virtual properties in a way
to make simple to handle in this algorithm and to extract .vtk files to visualize
the virtual trees using paraview. After these manipulations, the algorithm divide the space containing
the liver geometry into elementary volumes and then, for each one of them, it calls the function
EP3 which will browse the two trees, and compute the values of the multicompartment medium 
properties.   
"""

import numpy as np
from Fragment_volume import elementary_parametrization3D as EP3
#import random as rd
import matplotlib.pyplot as plt
from Export_DATA import Export_initial_from_xml as EI
from Export_DATA import Export_final_to_parametrization as EF


################# Define inputs and outputs paths ########################

CCO_path='CCO/'
Parameters_path='Parametrization/'
visualization_path='Visualization_tests/'

################# read and treat Portal Vein network ########################

PV_input_xml=CCO_path+'PV_3D_20000.xml'
PV_output_coordinates=CCO_path+'Trees_properties/PV_node_coordinates.txt'
PV_output_short_properties=CCO_path+'Trees_properties/PV_edge_short_attributes.txt'

EI(PV_input_xml,PV_output_coordinates,PV_output_short_properties)

PV_output_full_properties=CCO_path+'Trees_properties/PV_edge_full_attributes.txt'
PV_output_vtk=CCO_path+'VTK_visualization/PV_VTK_CCO.vtk'
PV_initial_pressure=1330.0

PV_Pmin=EF(PV_output_coordinates,PV_output_short_properties,PV_output_full_properties,PV_output_vtk,PV_initial_pressure)


PV_edges=np.loadtxt(PV_output_full_properties)
PV_vert=np.loadtxt(PV_output_coordinates)

PV_Connectivity=PV_edges[:,0:2]
PV_Connectivity=PV_Connectivity.astype(int)


################# read and treat Hepatic Vein network ########################


HV_input_xml=CCO_path +'HV_3D_20000.xml'
HV_output_coordinates=CCO_path+'Trees_properties/HV_node_coordinates.txt'
HV_output_short_properties=CCO_path+'Trees_properties/HV_edge_short_attributes.txt'

EI(HV_input_xml,HV_output_coordinates,HV_output_short_properties)

HV_output_full_properties=CCO_path+'Trees_properties/HV_edge_full_attributes.txt'
HV_output_vtk=CCO_path+'VTK_visualization/HV_VTK_CCO.vtk'
HV_initial_pressure=930.0

HV_Pmin=EF(HV_output_coordinates,HV_output_short_properties,HV_output_full_properties,HV_output_vtk,HV_initial_pressure)


HV_edges=np.loadtxt(HV_output_full_properties)
HV_vert=np.loadtxt(HV_output_coordinates)

HV_Connectivity=HV_edges[:,0:2]
HV_Connectivity=HV_Connectivity.astype(int)


################# Physical properties declaration ########################

mu=3.6e-2  #Pa.s
rho=1.06e-6 #Kg/mm3
P2mean=(PV_Pmin+HV_initial_pressure)/2.0 #Transfert pressure (lobular pressure)

################ Computing variable declaration ##########################

nbcomp=3 #nb_compartiments
ns=np.size(PV_edges[:,0]) #nb_segments
nv=np.size(PV_vert[:,0])   #nb_vertices

Network=np.zeros((ns,10,2),dtype=float) #segments (edges) properties
Domaine=np.array([[min(PV_vert[:,0])-10,max(PV_vert[:,0])+10],[min(PV_vert[:,1])-10,max(PV_vert[:,1])+10],[min(PV_vert[:,2])-10,max(PV_vert[:,2])+10]]) #Liver domain limits
nbe=np.array([25,17,15]) #nb domaine elements
step=np.array([(Domaine[0,1]-Domaine[0,0])/(nbe[0]),(Domaine[1,1]-Domaine[1,0])/(nbe[1]),(Domaine[2,1]-Domaine[2,0])/(nbe[2])]) #step
Origins=np.zeros((nbe[2]*nbe[1]*nbe[0],3),dtype=float) #origines elements
XXX=np.arange(Domaine[0,0]+step[0]/2, Domaine[0,1]+step[0]/2, step[0], dtype=float)
YYY=np.arange(Domaine[1,0]+step[1]/2, Domaine[1,1]+step[1]/2, step[1], dtype=float)
ZZZ=np.arange(Domaine[2,0]+step[2]/2, Domaine[2,1]+step[2]/2, step[2], dtype=float)


Ke=np.zeros((np.size(Origins[:,1]),9,nbcomp),dtype=float) #Elementary permeability matrix
Qe=np.zeros((np.size(Origins[:,1]),4,nbcomp),dtype=float) #Elementary flux matrix
Pe=np.zeros((np.size(Origins[:,1]),4,nbcomp),dtype=float) #Elementary pressure matrix
Beta=np.zeros((np.size(Origins[:,1]),4,nbcomp-1),dtype=float) #Elementary Coupling coefficient matrix
Intersept=np.zeros(np.size(Origins[:,1]),dtype=float) #number of segment intersepting an elementary volume



################# Variable initialisation and origins filling #####################

Ve=(step[0]*step[1]*step[2])*1e-9 #Elementary volumes
Pe[:,3,1]=P2mean

w=0
for i in range(nbe[2]):
    for j in range(nbe[1]):
        for k in range(nbe[0]):
            Origins[w,0]=Domaine[0,0]+(k+0.5)*step[0]
            Origins[w,1]=Domaine[1,0]+(j+0.5)*step[1]
            Origins[w,2]=Domaine[2,0]+(i+0.5)*step[2]
            w+=1
            



for i in range(nbcomp):
    for j in range(3):
        Ke[:,j,i]=Origins[:,j] 
        Qe[:,j,i]=Origins[:,j] 
        Pe[:,j,i]=Origins[:,j] 
        if i<nbcomp-1:
            Beta[:,j,i]=Origins[:,j] 
            


############# Network properties matrix filling: 0-> PV; 1 -> HV ####################
for i in range(ns):
            
        
    Network[i,0,0]=PV_vert[PV_Connectivity[i,0],0] # coordonnées x du point 1
    Network[i,1,0]=PV_vert[PV_Connectivity[i,0],1] # coordonnées y du point 1
    Network[i,2,0]=PV_vert[PV_Connectivity[i,0],2] # coordonnées z du point 1
    Network[i,3,0]=PV_vert[PV_Connectivity[i,1],0] # coordonnées x du point 2
    Network[i,4,0]=PV_vert[PV_Connectivity[i,1],1] # coordonnées y du point 2
    Network[i,5,0]=PV_vert[PV_Connectivity[i,1],2] # coordonnées z du point 2
    Network[i,6,0]=PV_edges[i,2] #inlet pressure
    Network[i,7,0]=PV_edges[i,3] #outlet pressure
    Network[i,8,0]=PV_edges[i,6]     #Flux (max=5*1e-5)
    Network[i,9,0]=2*PV_edges[i,4]  #diameter of the outlet point
    
    
    
    
    
    Network[i,0,1]=HV_vert[HV_Connectivity[i,0],0] # coordonnées x du point 1
    Network[i,1,1]=HV_vert[HV_Connectivity[i,0],1] # coordonnées y du point 1
    Network[i,2,1]=HV_vert[HV_Connectivity[i,0],2] # coordonnées z du point 1
    Network[i,3,1]=HV_vert[HV_Connectivity[i,1],0] # coordonnées x du point 2
    Network[i,4,1]=HV_vert[HV_Connectivity[i,1],1] # coordonnées y du point 2
    Network[i,5,1]=HV_vert[HV_Connectivity[i,1],2] # coordonnées z du point 2
    Network[i,6,1]=HV_edges[i,2] #inlet pressure
    Network[i,7,1]=HV_edges[i,3] #outlet pressure
    Network[i,8,1]=HV_edges[i,6]     #Flux (max=5*1e-5)
    Network[i,9,1]=2*HV_edges[i,4]  #diameter of the outlet point
    
    
    

        

####################### Parametrization loop ###################################
Avance=-0.01
for j in range(np.size(Origins[:,1])): #Browse the elementary volumes
    
    
    Avance_ac=(j/np.size(Origins[:,1]))*100
    if Avance_ac-Avance>=0.01:
        Avance+=0.01
        print('Avancement: ',round(Avance_ac,2),'%') #display progression

    Qj=np.zeros((nbcomp,),dtype=float) #Actual Element flux
    Pj=np.zeros((nbcomp,),dtype=float) #Actual Element pressure
    Kj=np.zeros((6,nbcomp,),dtype=float) #Actual Element permeability
    Vj=np.zeros((nbcomp,),dtype=float) #Branches volume in the actual element
    
    
    for i in range(ns): #Browse the Network segments
        
        for w in range(2): #Browse the two tress  0 -> PV, 1 -> HV
            V,l,d,s1,s2,s3=EP3(Origins[j,:],step/2,[Network[i,0,w],Network[i,1,w],Network[i,2,w]],[Network[i,3,w],Network[i,4,w],Network[i,5,w]],Network[i,9,w])
            if l!=0:
                Intersept[j]+=1
                if d<=(0.3*1e-3): 
                    NCP=1 #The compartment here is the filtration system
                    
                elif w==0:
                    NCP=0 #The compartment here is the Portal vein
                    if np.count_nonzero(PV_Connectivity[:,0] == PV_Connectivity[i,1])==0:
                        Qj[NCP]+=(Network[i,8,w])*1e-9 
                        Pj[NCP]+=V*(Network [i,6,w]+Network[i,7,w])/2
                        Vj[NCP]+=V
                    
                else:
                    NCP=2 #The compartment here is the Hepatic vein
                    if np.count_nonzero(HV_Connectivity[:,0] == HV_Connectivity[i,1])==0:
                        Qj[NCP]+=(Network[i,8,w])*1e-9 
                        Pj[NCP]+=V*(Network [i,6,w]+Network[i,7,w])/2
                        Vj[NCP]+=V
                
                Kj[0,NCP]+=(d**4*s1**2)/l #k11
                Kj[1,NCP]+=(d**4*s1*s2)/l #k12
                Kj[2,NCP]+=(d**4*s1*s3)/l #k13
                Kj[3,NCP]+=(d**4*s2**2)/l #k22
                Kj[4,NCP]+=(d**4*s2*s3)/l #k23
                Kj[5,NCP]+=(d**4*s3**2)/l #k33
                

            
    
    Qe[j,3,:]=Qj/Ve  # To the Global elementary flux matrix 
    for i in range(nbcomp):
        if Vj[i]!=0:
            Pe[j,3,i]=Pj[i]/Vj[i]  #To the Global elementary Pressure matrix 
    for k in range(6):
        Ke[j,k+3,:]=(np.pi/(128*mu*Ve))*Kj[k,:]  #To the Global elementary Permeability terms matrix
        


for i in range(np.size(Origins[:,1])):
    for j in range(nbcomp-1):
        if np.abs(Pe[i,3,j]-Pe[i,3,j+1])>0.01:
            Beta[i,3,j]=np.abs(Qe[i,3,j]-Qe[i,3,j+1])/np.abs(Pe[i,3,j]-Pe[i,3,j+1])   #To the Global Couplng coefficient matrix
        






############ Declaration and filling of the final properties matrices #################


K11C1=np.zeros((nbe[0]*nbe[1]*nbe[2],9),dtype=float)     ##### 1st Compartment Permeability 
K11C2=np.zeros((nbe[0]*nbe[1]*nbe[2],9),dtype=float)     ##### 2nd Compartment Permeability 
K11C3=np.zeros((nbe[0]*nbe[1]*nbe[2],9),dtype=float)     ##### 3rd Compartment Permeability 
G123=np.zeros((nbe[0]*nbe[1]*nbe[2],5),dtype=float)      ##### Coupling Coefficients


K11C1=Ke[:,:,0]
K11C2=Ke[:,:,1]
K11C3=Ke[:,:,2]
G123[:,0:4]=Beta[:,:,0]
G123[:,4]=Beta[:,3,1]


fmt = '%1.3f', '%1.3f', '%1.3f', '%.10e','%.10e','%.10e','%.10e','%.10e','%.10e'

np.savetxt(Parameters_path+'KeC1',K11C1,fmt=fmt)
np.savetxt(Parameters_path+'KeC2',K11C2,fmt=fmt)
np.savetxt(Parameters_path+'KeC3',K11C3,fmt=fmt)

fmt='%1.3f', '%1.3f', '%1.3f', '%.10e','%.10e'
np.savetxt(Parameters_path+'G123',G123,fmt=fmt)


np.savetxt(Parameters_path+'MESH',Origins,fmt='%1.5f')            ####### Parametrization GRID

      






###########################################################################################################

     #VISUALISATION TESTS
###########################################################################################################
ax=plt.axes(projection='3d')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
j=1
for i in range(ns):
    ax.plot3D([Network[i,0,j],Network[i,3,j]], [Network[i,1,j],Network[i,4,j]],[Network[i,2,j],Network[i,5,j]], linestyle='solid',color='k', marker="",markevery=0.01)    


ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
plt.savefig(visualization_path+'network3D0.png')        
        

#######################################################################
XX, YY = np.meshgrid(XXX, YYY)
RESL=np.reshape(Ke[2975:3400,3,0],   (np.size(YYY),np.size(XXX)))
fig, ax=plt.subplots()
fig.set_dpi(200)
fig.set_size_inches(8,6, forward=True)

plt.contourf(XX,YY,RESL)
plt.colorbar()

#plt.legend(loc='upper right');
#plt.grid(axis='both')
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
plt.title("Kxx, C1")

plt.savefig(visualization_path+'Ke11.png')
#######################################################################
RESL=np.reshape(Ke[2975:3400,6,1],   (np.size(YYY),np.size(XXX)))
fig, ax=plt.subplots()
fig.set_dpi(200)
fig.set_size_inches(8,6, forward=True)

plt.contourf(XX,YY,RESL)
plt.colorbar()

#plt.legend(loc='upper right');
#plt.grid(axis='both')
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
plt.title("Kyy, C2")

plt.savefig(visualization_path+'Ke22.png')
#######################################################################
RESL=np.reshape(Ke[2975:3400,8,2],   (np.size(YYY),np.size(XXX)))
fig, ax=plt.subplots()
fig.set_dpi(200)
fig.set_size_inches(8,6, forward=True)

plt.contourf(XX,YY,RESL)
plt.colorbar()

#plt.legend(loc='upper right');
#plt.grid(axis='both')
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
plt.title("Kzz, C3")

plt.savefig(visualization_path+'Ke33.png')
