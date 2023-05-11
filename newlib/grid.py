from numpy import array, ndarray, argmax, copy, ones, meshgrid
import matplotlib.pyplot as plt

class Slab:
  def __init__(self, lb:ndarray, ub:ndarray):
    self.lowerBounds = lb
    self.upperBounds = ub
  
  def get_range(self,idim):
    return array([self.lowerBounds[idim],self.upperBounds[idim]])

class IndexSlab:
  def __init__(self, nx):
    self.nx = nx
    self.ndim = len(nx)
  
  def getIndices(self, linInd):
    if (self.ndim == 1):
      return array([linInd])
    elif (self.ndim == 2):
      return array([linInd % self.nx[0], linInd // self.nx[0]])
    else:
      nxny = self.nx[0]*self.nx[1]
      return array([linInd % self.nx[0], linInd % nxny, linInd // nxny])

class Cell:
  hasGeom = False

  def __init__(self, inds:ndarray, pos:ndarray):
    self.position = pos
    self.indices = inds
  
  def setGeom(hgb):
    self.hasGeom = hgb
  
class Grid:
  numCells = ()
  ndims = ()
  dx = ()
  slab = None
  posSlab = None
  cells = dict()

  def __init__(self, numCells:tuple, startPos:tuple, endPos:tuple):
    if (len(numCells) != len(startPos) or \
        len(numCells) != len(endPos) or \
        len(startPos) != len(endPos)):
      raise(Exception("SLAGG error: specified grid dimensionality not consistent in startPos, endPos, and numCells"))
    self.numCells = array(numCells)
    self.ndims = len(numCells)
    self.dx = (array(endPos)-array(startPos)) / array(numCells)
    self.slab = Slab(array(0 for i in self.numCells),array(numCells))
    self.posSlab = Slab(array(startPos),array(endPos))
    self.lengths = array(endPos)-array(startPos)

    print("Initializing Grid with",numCells,"cells")

    # generate set of cells
    if (self.ndims == 1):
      for i in range(self.numCells[0]):
        self.cells[(i)] = Cell((i,),(startPos[0]+i*self.dx[0],))
    elif (self.ndims == 2):
      for i in range(self.numCells[0]):
        for j in range(self.numCells[1]):
          self.cells[(i,j)] = Cell((i,j),(startPos[0]+i*self.dx[0],startPos[1]+j*self.dx[1]))
    elif (self.ndims == 3):
      for i in range(self.numCells[0]):
        for j in range(self.numCells[1]):
          for k in range(self.numCells[2]):
            self.cells[(i,j,k)] = Cell((i,j,k),(startPos[0]+i*self.dx[0],startPos[1]+j*self.dx[1],startPos[2]+k*self.dx[2]))
    else:
      raise(Exception("SLAGG error: grids must be 1, 2, or 3-dimensional."))
    
  def getCell(self,inds:tuple):
    return self.cells[tuple(inds)]
  
  def getIndAtPos(self,pos):
    return array(pos)-self.posSlab.lowerBounds/array(self.lengths)*self.numCells

class Decomp:
  slabs = []
  nslabs = 1

  def __init__(self,grid,nslabs):
    self.nslabs = nslabs
    self.grid = grid
    
    # do regular decomposition
    self.__perform_regular_decomp()

  def __perform_regular_decomp(self):
    factors = self.__prime_factors(self.nslabs)
    print("prime factors: ", factors)

    domain_size = copy(self.grid.numCells)
    num_domains = array([1 for i in self.grid.numCells])
    for f in factors:
      ind = argmax(domain_size)
      domain_size[ind] /= f
      num_domains[ind] *= f

    print("domain sizes: ", domain_size)
    print("number of domains: ", num_domains)

    coord_map = IndexSlab(num_domains)
    for islab in range(self.nslabs):
      coords = coord_map.getIndices(islab)
      lb = ones(self.grid.ndims,dtype=int)
      ub = ones(self.grid.ndims,dtype=int)
      for idim in range(self.grid.ndims):
        lb[idim] = coords[idim] * domain_size[idim]
        ub[idim] = (coords[idim]+1) * domain_size[idim]
      self.slabs.append(Slab(lb,ub))
    
    print("Domain slabs:")
    for slab in self.slabs:
      print("lb: ",slab.lowerBounds,", ub: ",slab.upperBounds)

  def plot(self,axes=None):
    if (self.grid.ndims == 3):
      if not axes:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_aspect("equal")

      for slab in self.slabs:
        self.__draw_3D_box(ax,slab)
    
    elif (self.grid.ndims == 2):
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      
      for slab in self.slabs:
        self.__draw_2D_box(ax,slab)
      
    else:
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      
      for slab in self.slabs:
        self.__draw_1D_box(ax,slab)
    
    plt.show()
  
  def __draw_3D_box(self,ax,slab):
    x_range = slab.get_range(0)
    y_range = slab.get_range(1)
    z_range = slab.get_range(2)

    xx, yy = meshgrid(x_range, y_range)
    zz0 = array([[z_range[0],z_range[0]],[z_range[0],z_range[0]]])
    zz1 = array([[z_range[1],z_range[1]],[z_range[1],z_range[1]]])
    ax.plot_wireframe(xx, yy, zz0, color="r")
    #ax.plot_surface(xx, yy, zz0, color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy, zz1, color="r")
    #ax.plot_surface(xx, yy, zz1, color="r", alpha=0.2)

    yy, zz = meshgrid(y_range, z_range)
    xx0 = array([[x_range[0],x_range[0]],[x_range[0],x_range[0]]])
    xx1 = array([[x_range[1],x_range[1]],[x_range[1],x_range[1]]])
    ax.plot_wireframe(xx0, yy, zz, color="r")
    #ax.plot_surface(xx0, yy, zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx1, yy, zz, color="r")
    #ax.plot_surface(xx1, yy, zz, color="r", alpha=0.2)

    yy0 = array([[y_range[0],y_range[0]],[y_range[0],y_range[0]]])
    yy1 = array([[y_range[1],y_range[1]],[y_range[1],y_range[1]]])
    ax.plot_wireframe(xx, yy0, zz, color="r")
    #ax.plot_surface(xx, yy0, zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy1, zz, color="r")
    #ax.plot_surface(xx, yy1, zz, color="r", alpha=0.2)
  
  def __draw_2D_box(self,ax,slab):
    x_range = slab.get_range(0)
    y_range = slab.get_range(1)

    xx, yy = meshgrid(x_range, y_range)
    ax.plot(xx[0], yy[0], color="r")
    ax.plot(xx[0], yy[1], color="r")
    ax.plot(xx[1], yy[0], color="r")
    ax.plot(xx[1], yy[1], color="r")
  
  def __draw_1D_box(self,ax,slab):
    x_range = slab.get_range(0)

    xx, yy = meshgrid(x_range, array([-1,1]))
    ax.plot(xx[0], yy[0], color="r")
    ax.plot(xx[0], yy[1], color="r")
    ax.plot(xx[1], yy[0], color="r")
    ax.plot(xx[1], yy[1], color="r")
  
  def __prime_factors(self, n):
    i = 2
    factors = []
    while i * i <= n:
      if n % i:
        i += 1
      else:
        n //= i
        factors.append(i)
    if n > 1:
      factors.append(n)
    return factors