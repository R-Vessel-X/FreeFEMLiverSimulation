# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:42:05 2023

@author: M.Chetoui
"""
import lxml.etree as ET
import vtk
import numpy as np


mu=3.6e-2  #Blood viscosity in Pa.s

def Export_initial_from_xml(input_xml,output_coordinates,output_properties):
    
    # Parse the XML data
    tree = ET.parse(input_xml)
    root = tree.getroot()
    k=-1
    # Create a file to store the coordinates
    output_file = open(output_coordinates, 'w')
    
    # Iterate over the nodes and extract the coordinates
    for node in root.findall('.//node'):
        node_id = node.get('id')[1:]
        position = node.find('.//attr[@name=" position"]/tup')
        coordinates = position.findall('float')
    
        # Extract x, y, and z coordinates
        x = str(float(coordinates[0].text))
        z = str(float(coordinates[1].text))
        y = str(-float(coordinates[2].text))
        k+=1
        
    
        # Write the coordinates to the file
        output_file.write(x+" "+y+" "+z+" \n")
    
    # Close the output file
    output_file.close()
    
    # Create a file to store the edge attributes
    output_file = open(output_properties, 'w')
    
    # Iterate over the edges and extract the attributes
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')[1:]
        to_node = edge.get('to')[1:]
        from_node = edge.get('from')[1:]
    
        flow = edge.find('.//attr[@name=" flow"]/float').text
        resistance = edge.find('.//attr[@name=" resistance"]/float').text
        radius = edge.find('.//attr[@name=" radius"]/float').text
    
        # Write the attributes to the file
        output_file.write(f"{from_node} {to_node} {radius} {flow} {resistance}\n")
    
    # Close the output file
    output_file.close()
    
    
def Export_final_to_parametrization(input_coordinates,input_properties,output_properties,output_vtk,initial_pressure):

    PV_edges=np.loadtxt(input_properties)
    PV_vert=np.loadtxt(input_coordinates)
    ns=np.size(PV_edges[:,0]) #nb_branches
    nv=np.size(PV_vert[:,0])   #nb_vertices

    Connectivity=PV_edges[:,0:2]
    Connectivity=Connectivity.astype(int)
    PV_rad=PV_edges[:,2]
    PV_rad=PV_rad.astype(float)
    PV_rad=PV_rad/4.65**(0.25)  #corrected radius
    
    Pressure=np.zeros((nv,), dtype=float)  #Pressure in the vertices
    PV_len=np.zeros((ns,), dtype=float)  #lenths of the segments
    PV_res=np.zeros((ns,), dtype=float)  #resistance of the segments vertices
    PV_vel=np.zeros((ns,), dtype=float)  #velocity of the segments vertices
    Pressure[0]=initial_pressure

    nz=0

    for i in range(ns):
        PV_len[i]=np.sqrt((PV_vert[Connectivity[i,0],0]-PV_vert[Connectivity[i,1],0])**2+(PV_vert[Connectivity[i,0],1]-PV_vert[Connectivity[i,1],1])**2+(PV_vert[Connectivity[i,0],2]-PV_vert[Connectivity[i,1],2])**2)
        PV_res[i]=((8*mu*PV_len[i])/(np.pi*PV_rad[i]**4))
        PV_vel[i]=PV_edges[i,3]/(np.pi*PV_rad[i]**2)

    while nz<(ns-1):
        for i in range(ns):
            if np.abs(Pressure[Connectivity[i,1]])<0.001 and np.abs(Pressure[Connectivity[i,0]])>0.001:
                
                Pressure[Connectivity[i,1]]=Pressure[Connectivity[i,0]]-(PV_res[i]*PV_edges[i,3])
                nz+=1
        print(nz, ' pressures computed')
        
    # Create a file to store the edge attributes    
    output_file = open(output_properties, 'w')
    
    
    # Iterate over the edges and extract the attributes n1 n2 p1 p2 radius length flux velocity resistance
    for i in range(ns):
        output_file.write(str(Connectivity[i,0])+" "+str(Connectivity[i,1])+" "+str(Pressure[Connectivity[i,0]])+" "+str(Pressure[Connectivity[i,1]])+" "+str(PV_rad[i])+" "+str(PV_len[i])+" "+str(PV_edges[i,3])+" "+str(PV_vel[i])+" "+ str(PV_res[i])+" \n")
        
    
    # Close the output file
    output_file.close()
    
    # Create VTK points and cells
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    # Create VTK data arrays to store radius and thickness
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    thickness_array = vtk.vtkFloatArray()
    thickness_array.SetName("Thickness")
    flow_array = vtk.vtkFloatArray()
    flow_array.SetName("Flow")
    velocity_array = vtk.vtkFloatArray()
    velocity_array.SetName("Velocity")
    length_array = vtk.vtkFloatArray()
    length_array.SetName("Length")
    resistance_array = vtk.vtkFloatArray()
    resistance_array.SetName("Resistance")
    Pressure_array = vtk.vtkFloatArray()
    Pressure_array.SetName("Mean pressure")
    
    # Iterate over the nodes and add points
    for i in range(nv):
    
        # Extract x, y, and z coordinates
        x = PV_vert[i,0]
        y = PV_vert[i,1]
        z = PV_vert[i,2]
        points.InsertNextPoint(x, y, z)
        
        
        
        
    # Iterate over the edges and add lines
    for i in range(ns):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, Connectivity[i,0])
        line.GetPointIds().SetId(1, Connectivity[i,1])
        lines.InsertNextCell(line)
        
        # Extract the properties from the edge attributes
        radius = PV_rad[i]
        flow = PV_edges[i,3]
        velocity=PV_vel[i]
        pr=0.5*(Pressure[Connectivity[i,0]]+Pressure[Connectivity[i,1]])
        resistance=PV_res[i]
        length=PV_len[i]
    
        # Add the radius to the radius array
        radius_array.InsertNextValue(radius)
        flow_array.InsertNextValue(flow)
        velocity_array.InsertNextValue(velocity)
        Pressure_array.InsertNextValue(pr)
        resistance_array.InsertNextValue(resistance)
        length_array.InsertNextValue(length)
        
        
    # Calculate thickness based on the radius
    min_radius = radius_array.GetRange()[0]
    max_radius = radius_array.GetRange()[1]
    thickness_range = (0.1, 1.0)  # Define the desired range of thickness
    thickness_array.SetNumberOfValues(radius_array.GetNumberOfValues())
    for i in range(radius_array.GetNumberOfValues()):
        normalized_radius = (radius_array.GetValue(i) - min_radius) / (max_radius - min_radius)
        thickness = thickness_range[0] + (thickness_range[1] - thickness_range[0]) * normalized_radius
        thickness_array.SetValue(i, thickness)
        
        
    # Create a polydata object and add points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    # Add the thickness array to the polydata
    polydata.GetCellData().AddArray(radius_array)
    polydata.GetCellData().AddArray(flow_array)
    polydata.GetCellData().AddArray(flow_array)
    polydata.GetCellData().AddArray(velocity_array)
    polydata.GetCellData().AddArray(Pressure_array)
    polydata.GetCellData().AddArray(resistance_array)
    polydata.GetCellData().AddArray(length_array)
    polydata.GetCellData().AddArray(thickness_array)
    
    # Apply the vtkTubeFilter to create tubular representations with varying thickness
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(polydata)
    tube_filter.SetRadius(0.05)  # Set a fixed radius for the tubes
    tube_filter.SetNumberOfSides(20)  # Set the number of sides for the tubes
    tube_filter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Thickness")
    tube_filter.Update()
    
    # Write the polydata to a VTK file
    vtk_writer = vtk.vtkPolyDataWriter()
    vtk_writer.SetFileName(output_vtk)
    vtk_writer.SetInputData(tube_filter.GetOutput())
    vtk_writer.Write()
    
    return(min(Pressure))
