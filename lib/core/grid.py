from OCC.Core.BRepPrimAPI import *
from OCC.Core.BRepTools import *
from OCC.Core.gp import *
from OCC.Extend.TopologyUtils import *

import matplotlib.path as mpl_path
import numpy as np

from core import cell, geometry


class Grid:

  # Assume uniform scale and rectangular array of cells
  scale = 1.0
  dims = []
  shape = None
  cells = []
  nudge = []
  
  def __init__(self, s, d, geo, nudge):
    self.dims = d
    self.scale = float(s)
    self.shape = geo
    self.nudge = nudge
    print(nudge)
    
    for i in range(self.dims[0]):
      for j in range(self.dims[1]):
        for k in range(self.dims[2]):
          # Initialize a cell to the grid, then see if it contains geometry
          new_cell = cell.Cell([i, j, k], self.scale)

          # For algorithm testing (and for basic 5x5, or for complex 10x5 + 5x5)
          # if (i < 5) or (j < 5):
          #   new_cell.set_geo_cell(True)
        
          self.cells.append(new_cell)    

    # Start by flagging each cell with a vertex in it as a geometry cell
    brt = BRep_Tool()
    topo = TopologyExplorer(self.shape)
    verts = topo.vertices()
  
    def __conv_to_inds(realx, realy, realz):
      return [int((realx + self.nudge[0]) // self.scale), int((realy + self.nudge[1]) // self.scale), int((realz + self.nudge[2]) // self.scale)]
  
    for v in verts:
      pt = brt.Pnt(v)
      geo_cell_inds = __conv_to_inds(pt.X(), pt.Y(), pt.Z())
      if (geo_cell_inds[0] < self.dims[0] and geo_cell_inds[1] < self.dims[1] and geo_cell_inds[2] < self.dims[2]):
        self.get_cell(geo_cell_inds[0], geo_cell_inds[1], geo_cell_inds[2]).set_geo_cell(True)
      
    for k in range(self.dims[2]):
      # Scan through each row and flag the cells in bewteen as geo cells
      for i in range(self.dims[0]):
        num_vert_cells = 0
        start_col = 0
        end_col = 0
        for j in range(self.dims[1]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_col = j
            elif num_vert_cells >= 2:
              end_col = j
        if num_vert_cells >= 2:
          for j in range(start_col, end_col + 1):
            self.get_cell(i, j, k).set_geo_cell(True)
      # And now columns
      for j in range(self.dims[1]):
        num_vert_cells = 0
        start_row = 0
        end_row = 0
        for i in range(self.dims[0]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_row = i
            elif num_vert_cells >= 2:
              end_row = i
        if num_vert_cells >= 2:
          for i in range(start_row, end_row + 1):
            self.get_cell(i, j, k).set_geo_cell(True)            

    for j in range(self.dims[1]):
      # Scan through each row and flag the cells in bewteen as geo cells
      for i in range(self.dims[0]):
        num_vert_cells = 0
        start_layer = 0
        end_layer = 0
        for k in range(self.dims[2]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_layer = k
            elif num_vert_cells >= 2:
              end_layer = k
        if num_vert_cells >= 2:
          for k in range(start_layer, end_layer + 1):
            self.get_cell(i, j, k).set_geo_cell(True)
      # And now columns
      for k in range(self.dims[2]):
        num_vert_cells = 0
        start_row = 0
        end_row = 0
        for i in range(self.dims[0]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_row = i
            elif num_vert_cells >= 2:
              end_row = i
        if num_vert_cells >= 2:
          for i in range(start_row, end_row + 1):
            self.get_cell(i, j, k).set_geo_cell(True)

    for i in range(self.dims[0]):
      # Scan through each row and flag the cells in bewteen as geo cells
      for j in range(self.dims[1]):
        num_vert_cells = 0
        start_layer = 0
        end_layer = 0
        for k in range(self.dims[2]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_layer = k
            elif num_vert_cells >= 2:
              end_layer = k
        if num_vert_cells >= 2:
          for k in range(start_layer, end_layer + 1):
            self.get_cell(i, j, k).set_geo_cell(True)
      # And now columns
      for k in range(self.dims[2]):
        num_vert_cells = 0
        start_col = 0
        end_col = 0
        for j in range(self.dims[1]):
          if self.get_cell(i, j, k).has_geometry():
            num_vert_cells = num_vert_cells + 1
            if num_vert_cells == 1:
              start_col = j
            elif num_vert_cells >= 2:
              end_col = j
        if num_vert_cells >= 2:
          for j in range(start_col, end_col + 1):
            self.get_cell(i, j, k).set_geo_cell(True)

  def get_cell(self, i, j, k):
    return self.cells[int(i * int(self.dims[1]) * int(self.dims[2]) + j * int(self.dims[2]) + k)]
    
  def get_nudge(self):
    return self.nudge
  
  def get_cells(self):
    return self.cells
    
  def get_dims(self):
    return self.dims

  def get_size(self):
    return int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])

  def get_scale(self):
    return self.scale  
