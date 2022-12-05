import matplotlib.path as mpl_path
import numpy as np

from core import cell, geometry


class Grid:

  # Assume uniform scale and rectangular array of cells
  scale = 1.0
  dims = []
  shape = None
  cells = []
  
  def __init__(self, s, d, geo):
    self.dims = d
    self.scale = s
    self.shape = geo
    verticies = []
    # verticies = np.array(shape.get_verticies())
    # geo_path = mpl_path.Path(verticies)
    
    for i in range(self.dims[0]):
      for j in range(self.dims[1]):
        for k in range(self.dims[2]):
          # Initialize a cell to the grid, then see if it contains geometry
          new_cell = cell.Cell([i, j, k], self.scale)
          
          # For algorithm testing
          if (i < 5) or (j < 5):
            new_cell.set_geo_cell(True)
        
          # Cell contains geometry iff the geometry intersects one of its edges
          # or if an edge is wholly within the geometry bounds
          lowers = new_cell.real_bounds_low()
          uppers = new_cell.real_bounds_high()
          # bot_left_pt = (float(lowers[0]), float(lowers[1]))
          # bot_right_pt = (float(lowers[0]), float(uppers[1]))
          # top_left_pt = (float(lowers[1]), float(lowers[0]))
          # top_right_pt = (float(lowers[1]), float(uppers[1]))
        
          # top_edge = mpl_path.Path(np.array([top_left_pt, top_right_pt]))
          # right_edge = mpl_path.Path(np.array([bot_rightt_pt, top_right_pt]))
          # bot_edge = mpl_path.Path(np.array([bot_left_pt, bot_right_pt]))
          # left_edge = mpl_path.Path(np.array([bot_left_pt, bot_right_pt]))
          # for edge in [left_edge, right_edge, bot_edge, top_edge]:
          #   if geo_path.intersects_path(edge) or geo_path.contains_path(edge):
          #     new_cell.set_geo_cell(True)
          #     break
        self.cells.append(new_cell)    

  def get_cell(self, i, j, k):
    return self.cells[i * self.dims[1] * self.dims[2] + j * self.dims[2] + k]
  
  def get_cells(self):
    return self.cells
    
  def get_dims(self):
    return self.dims

  def get_size(self):
    return int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])
