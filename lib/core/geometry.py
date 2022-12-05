from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
from OCC.Core.StlAPI import StlAPI_Writer, StlAPI_Reader
from OCC.Display.WebGl import x3dom_renderer

import os

class Geometry:

  filename = None
  myshape = None
  
  def __init__(self, filename):
    self.filename = filename

  def read_cad(self):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(self.filename)
    step_reader.TransferRoot()
    self.myshape = step_reader.Shape()
    print(type(self.myshape))
    return self.myshape
    
  def translate_to_step(self, reload):
    stl_reader = StlAPI_Reader()
    stl_reader.SetASCIIMode(True)
    stl_reader.ReadFile(filename)
    temp_shape = stl_reader.Shape()

    outname = os.path.basename(filename) + '.step'  
    step_writer = STEPControl_Writer()
    step_writer.Write(temp_shape, outname)
    if reload:
      self.set_filename(outname)
      self.read_cad(self.filename)
      
  def visualize(self):
    my_renderer = x3dom_renderer.X3DomRenderer()
    my_renderer.DisplayShape(self.myshape)
    my_renderer.render()
    
  def set_filename(self, new_name):
    self.filename = new_name