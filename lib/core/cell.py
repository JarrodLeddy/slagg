class Cell:

  inds = []
  scale = []
  geo_cell = False
  # All cells are 3D

  def __init__(self, idx, scaling):
    self.inds = idx
    self.scale = scaling
    self.dim = len(self.inds)
    assert self.dim == 3, 'Cell in 3D space initialized with fewer than three dimensions'
    
  def int_bounds(self):
    return self.inds

  def real_bounds_low(self):
    coords = []
    for i in range(self.dim):
      # idx: index array of cell
      # scale: scales of each individual cell (assume uniformity)
      coords.append(float(self.inds[i]) * float(self.scale[i]))
    return coords
    
  def real_bounds_high(self):
    coords = []
    for i in range(self.dim):
      # idx: index array of cell
      # scale: scales of each individual cell (assume uniformity)
      coords.append(float(self.inds[i] + 1) * float(self.scale[i]))
    return coords
    
  def has_geometry(self):
    return self.geo_cell
  
  # geo_cell just determines if the cell has geometry in it or not;
  # gets set with grid initialization  
  def set_geo_cell(self, has_geo):
    self.geo_cell = has_geo
