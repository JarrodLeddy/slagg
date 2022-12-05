import math
import matplotlib.pyplot as plt
import numpy as np

from core import cell, grid

class Decomp:
  start_cell = None
  end_cell = None
  grid = None
  dim = 3
  
  def __init__(self, mygrid, start, end):
    # Define the panel with two cells: lowest indexed cell (start) and highest
    # (end)
    self.start_cell = start
    self.end_cell = end
    self.grid = mygrid
    self.dim = len(self.start_cell.int_bounds())

  def num_cells(self):
    # Assume rectagular grid, so num_cells is just product of each dimension size
    return math.prod(self.dims)
    
  def has_vacancies(self):
    start_inds = self.start_cell.int_bounds()
    end_inds = self.end_cell.int_bounds()
    
    for i in range(start_inds[0], end_inds[0]):
      for j in range(start_inds[1], end_inds[1]):
        for k in range(start_inds[2], end_inds[2]):
          if not self.grid.get_cell(i, j, k).has_geometry():
            return False
    return True
    
  def expand(self, vec):
    # Expand the panel left and then up by specified number of cells in each
    # direction
    self.start_cell = start_cell.shift(-1 * vec[0])
    self.end_cell = end_cell.shift(vec[1])
    
  def shift(self, vec):
    # Move the panel along vec
    self.start_cell = start_cell.shift(vec)
    self.end_cell = end_cell.shift(vec)
  
  def get_start_cell(self):
    return self.start_cell
    
  def get_end_cell(self):
    return self.end_cell
    
  def validate(self):
    for i in range(dim):
      if self.start_cell.int_bounds[i] > self.end_cell.int_bounds[i]:
        return False
    return True
    
  def contains_real(self, pos):
    for i in range(dim):
      if pos[i] < self.start_cell.real_bounds_low[i] or pos[i] >= self.end_cell.real_bounds_high[i]:
        return False
    return True
    
  def contains_inds(self, inds):
     for i in range(dim):
      if pos[i] < self.start_cell.ind_bounds[i] or pos[i] >= self.end_cell.ind_bounds[i]:
        return False
     return True
    
  def length(self, i):
    # Return length of rectangle in the i direction
    return self.end_cell.int_bounds()[i] - self.start_cell.int_bounds()[i] + 1
    
  def draw(self, ax):
    # Lower rectangle
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    # Upper rectangle
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    # Connect them
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1], self.start_cell.real_bounds_low()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0], self.start_cell.real_bounds_low()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.start_cell.real_bounds_low()[1], self.start_cell.real_bounds_low()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    edge1x = [self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0], self.end_cell.real_bounds_high()[0] - self.grid.get_nudge()[0]]
    edge1y = [self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1], self.end_cell.real_bounds_high()[1] - self.grid.get_nudge()[1]]
    edge1z = [self.start_cell.real_bounds_low()[2] - self.grid.get_nudge()[2], self.end_cell.real_bounds_high()[2] - self.grid.get_nudge()[2]]
    ax.plot3D(edge1x, edge1y, edge1z, color='red')
    
    # And don't ask me to do that again
                 