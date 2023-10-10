#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:36:01 2023

@author: mohamed
"""

import numpy as np
import pyvista as pv

# Load the data from the text file
data = np.loadtxt('Ke1.txt')

# Extract the coordinates and properties from the data
coordinates = data[:, 0:3]
properties = data[:, 3:]

# Create a PyVista point cloud with the coordinates
point_cloud = pv.PolyData(coordinates)

# Create a 3D mesh using the Delaunay3D filter
mesh = point_cloud.delaunay_3d()

# Get the cell connectivity information
connectivity = mesh.cells.reshape(-1, 5)[:, 1:]

# Initialize an array for cell properties
cell_properties = np.zeros((mesh.n_cells, properties.shape[1]))

# Interpolate the point data to the cells
for i, cell in enumerate(connectivity):
    # Get the indices of the cell's points
    cell_point_ids = cell

    # Get the properties of the cell's points
    cell_point_properties = properties[cell_point_ids]

    # Compute the mean of the point properties
    cell_property = np.mean(cell_point_properties, axis=0)

    # Assign the cell property
    cell_properties[i] = cell_property

# Add the interpolated properties as cell data to the mesh
for i in range(properties.shape[1]):
    mesh.cell_data[f'Property{i+1}'] = cell_properties[:, i]

# Save the mesh as a VTK file
mesh.save('Ke1.vtk')








