'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkWindowToImageFilter
)
from vtkmodules.vtkIOImage import (
    vtkPNGWriter,
)

import PIL.Image as Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras.preprocessing.image as kpi

def read_cell_num(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    polydata = reader.GetOutput()
    return polydata.GetNumberOfCells()

def create_png_from_vtk(filename):
    colors = vtkNamedColors()
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('White'))
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d('Black'))
    renWin.Render()
    writer = vtkPNGWriter()
    windowto_image_filter = vtkWindowToImageFilter()
    windowto_image_filter.SetInput(renWin)
    windowto_image_filter.SetScale(1)  # image quality
    windowto_image_filter.SetInputBufferTypeToRGB()
    windowto_image_filter.ReadFrontBufferOff()
    windowto_image_filter.Update()
    writer.SetFileName(filename+".png")
    writer.SetInputConnection(windowto_image_filter.GetOutputPort())
    writer.Write()

def create_pil_image(filename,delete=True):
    create_png_from_vtk(filename)
    image = Image.open(filename+".png").convert("RGB")
    if delete:
        if os.path.exists(filename+".png"):
            os.remove(filename+".png")
        else:
            print("Error in creating png image")
            exit(-99)
    return image

def pil_to_array(image):
    return np.asarray(image).copy()

def array_to_pil(array,width=300,height=300):
    return Image.fromarray(np.squeeze(array), 'RGB').copy().resize([width,height])

def keras_obs(image,width=300,height=300,channels=3):
    return kpi.img_to_array(image).reshape((1,width,height,channels))

def create_decentered_pil_image(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    polydata = reader.GetOutput()
    numpy_array_of_points = dsa.WrapDataObject(polydata).Points
    area = np.asarray(Image.new("RGB", (400,400), (0,0,0))).copy()
    for p in numpy_array_of_points:
        area[int(p[0])][int(p[1])] = [255,0,0]
    image = Image.fromarray(np.squeeze(area), 'RGB').copy().resize([400,400])
    return image

def add_target(array, target_center, target_radius, width=400, height=400):
    for x in range(width):
        for y in range(height):
            dist = (x-target_center[0])**2 + (y-target_center[1])**2 - target_radius**2
            if dist>-2*target_radius and dist<2*target_radius:
                array[x][y] = [255,255,255]
    return array

def count_target_points(filename, target_center, target_radius):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    polydata = reader.GetOutput()
    wrap_data = dsa.WrapDataObject(polydata)
    numpy_array_of_points = wrap_data.Points
    cellNPoints = wrap_data.GetCellData()['cellNPoints']
    inside = 0.0
    outside = 0.0
    up_to=0
    for c in cellNPoints:
        temp_inside = 0
        temp_outside = 0
        for p in numpy_array_of_points[up_to:up_to+c]:
            if (p[0]-target_center[0])**2 + (p[1]-target_center[1])**2 >= target_radius**2:
                temp_outside = temp_outside+1
            else:
                temp_inside = temp_inside+1
        up_to = up_to+c
        inside = inside + temp_inside/c
        outside = outside + temp_outside/c
    return outside, inside