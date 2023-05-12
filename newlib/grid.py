from numpy import array, ndarray, argmax, copy, ones, meshgrid, min, max, floor, append, mgrid, cross, dot
import matplotlib.pyplot as plt
from stl import mesh
import logging, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger('requests')
requests_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
requests_logger.addHandler(handler)

class Slab:
  def __init__(self, lb:ndarray, ub:ndarray):
    self.lowerBounds = array(lb)
    self.upperBounds = array(ub)
  
  def get_range(self,idim):
    return array([self.lowerBounds[idim],self.upperBounds[idim]])

class IndexSlab:
  def __init__(self, nx):
    self.nx = array(nx)
    self.ndim = len(nx)
  
  def getIndices(self, linInd):
    if (self.ndim == 1):
      return array([linInd])
    elif (self.ndim == 2):
      return array([linInd % self.nx[0], linInd // self.nx[0]])
    else:
      nxny = self.nx[0]*self.nx[1]
      return array([linInd % self.nx[0], (linInd % nxny) // self.nx[0], linInd // nxny])

class Cell:
  has_geometry = False

  def __init__(self, inds:ndarray, pos:ndarray, dx, contains_geometry=False):
    self.position = array(pos)
    self.indices = array(inds)
    self.slab = Slab(self.indices,self.indices+1)
    self.dx = dx
    self.has_geometry = contains_geometry
  
  def set_has_geometry(self, hgb):
    self.has_geometry = hgb
  
  def get_center(self):
    return self.position + 0.5*self.dx
  
class Grid:
  numCells = ()
  ndims = ()
  dx = ()
  slab = None
  posSlab = None
  cells = dict()
  geometry = None

  def __init__(self, numCells:tuple, startPos=None, endPos=None, geometry=None):

    self.numCells = array(numCells)
    self.ndims = len(numCells)

    # check if geometry defined, if not then endPos and startPos must be
    if ((startPos is None or endPos is None) and (geometry is None)):
      raise(Exception("SLAGG Grid error: Either geometry must be specified or start/end positions"))
    elif (geometry is not None):
      self.geometry = geometry
      verts = geometry.get_vertices()
      sp = ones(self.ndims)
      ep = ones(self.ndims)

      # find min and max of geometry, set startPos and endPos there
      for i in range(self.ndims):
        sp[i] = min(verts[:,i])
        ep[i] = max(verts[:,i])

      # debug output so the user can see the geometry loaded correctly
      logger.debug("Found geometry bounds:")
      logger.debug("start positions:  "+str(sp))
      logger.debug("end positions:    "+str(ep)+"\n")

      # now shift, add normalized padding, and shift back
      osp = copy(sp)
      lengths = array(ep)-array(sp)
      ep -= sp
      sp -= sp
      startPos = (sp - 0.05*lengths) + osp
      endPos = (ep + 0.05*lengths) + osp

      # tell the user what grid bounds were chosen
      logger.info("Using geometry to determine the grid size:")
      logger.info("start positions:  "+str(startPos))
      logger.info("end positions:    "+str(endPos)+"\n")

    else:
      if (len(numCells) != len(startPos) or \
          len(numCells) != len(endPos) or \
          len(startPos) != len(endPos)):
        raise(Exception("SLAGG Grid error: specified grid dimensionality not consistent in startPos, endPos, and numCells"))

    self.dx = (array(endPos)-array(startPos)) / array(numCells)
    self.slab = Slab(array([0 for i in self.numCells]),array(numCells))
    self.posSlab = Slab(array(startPos),array(endPos))
    self.lengths = array(endPos)-array(startPos)

    logger.info("Initializing Grid with "+str(numCells)+" cells")

    # generate set of cells
    if (self.ndims == 1):
      for i in range(self.numCells[0]):
        self.cells[(i)] = Cell((i,),(startPos[0]+i*self.dx[0],),self.dx)
    elif (self.ndims == 2):
      for i in range(self.numCells[0]):
        for j in range(self.numCells[1]):
          self.cells[(i,j)] = Cell((i,j),(startPos[0]+i*self.dx[0],startPos[1]+j*self.dx[1]),self.dx)
    elif (self.ndims == 3):
      for i in range(self.numCells[0]):
        for j in range(self.numCells[1]):
          for k in range(self.numCells[2]):
            self.cells[(i,j,k)] = Cell((i,j,k),(startPos[0]+i*self.dx[0],startPos[1]+j*self.dx[1],startPos[2]+k*self.dx[2]),self.dx)
    else:
      raise(Exception("SLAGG error: grids must be 1, 2, or 3-dimensional."))
    
    # set geometry flag for every cell that contains a vertex
    if (self.geometry is not None):
      verts = geometry.get_vertices()
      for v in verts:
        self.cells[tuple(self.get_ind_at_pos(v,True))].set_has_geometry(True)
    
      # now we need to see which cells have trianges intersect 
      # them since they contain geometry
      triangles = geometry.get_triangles()
      for t in triangles:
        p1,p2,p3 = [t[0:3],t[3:6],t[6:9]]
        for c in self.cells.values():
          #########
          # first check if plane of triangle intersects cell, if not then we are done
          # shift cube and triangle such that the cube is centered at 0,0,0
          shift = c.get_center()
          p1 -= shift
          p2 -= shift
          p3 -= shift
          c.set_has_geometry(self.geometry.check_intersection(p1,p2,p3,self.dx/2))
          p1 += shift
          p2 += shift
          p3 += shift
    
  def get_cell(self, inds:tuple):
    return self.cells[tuple(inds)]
  
  def get_ind_at_pos(self, pos, round=False):
    if not round:
      return (array(pos)-self.posSlab.lowerBounds)/ \
          array(self.lengths)*self.numCells + self.slab.lowerBounds
    return array(floor((array(pos)-self.posSlab.lowerBounds)/ \
        array(self.lengths)*self.numCells),dtype=int) + self.slab.lowerBounds
  
  def get_pos_at_ind(self, ind):
    return (array(ind)-array(self.slab.lowerBounds)) / \
        array(self.numCells)*self.lengths + self.posSlab.lowerBounds
  
  def set_geometry(self, geometry):
    return
  
  def plot(self, axes=None, plot=False, rectangles=False, geometry_only=True):
    if (self.ndims == 3):
      if not axes:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_aspect("equal")
      else:
        ax = axes

      for cell in self.cells.values():
        if (cell.has_geometry and geometry_only) or (not geometry_only):
          if (rectangles):
            PlotRectangles.draw_3D_box(ax,cell.slab)
          else:
            ax.scatter(cell.position[0], cell.position[1], cell.position[2], \
                marker='.', c='k')

      # equal aspect not gauranteed in 3D, make bounding box to plot
      max_range = self.lengths.max()
      Xb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() \
          + 0.5*self.posSlab.get_range(0).sum()
      Yb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() \
          + 0.5*self.posSlab.get_range(1).sum()
      Zb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() \
          + 0.5*self.posSlab.get_range(2).sum()
      for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    elif (self.ndims == 2):
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      else:
        ax = axes
      
      for cell in self.cells.values():
        if (cell.has_geometry and geometry_only) or (not geometry_only):
          if (rectangles):
            PlotRectangles.draw_2D_box(ax,cell.slab)
          else:
            ax.scatter(cell.position[0], cell.position[1], marker='.', c='k')
      
    else:
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      else:
        ax = axes
      
      for cell in self.cells.values():
        if (cell.has_geometry and geometry_only) or (not geometry_only):
          if (rectangles):
            PlotRectangles.draw_1D_box(ax,cell.slab)
          else:
            ax.scatter(cell.position[0], marker='.', c='k')

    if (plot):
      plt.show()
    
    return ax

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
    logger.debug(str(self.nslabs)+' slabs broken into prime factors: '+str(factors))

    domain_size = copy(self.grid.numCells)
    num_domains = array([1 for i in self.grid.numCells])
    for f in factors:
      ind = argmax(domain_size)
      domain_size[ind] /= f
      num_domains[ind] *= f

    self.coord_map = IndexSlab(num_domains)
    for islab in range(self.nslabs):
      coords = self.coord_map.getIndices(islab)
      lb = ones(self.grid.ndims,dtype=int)
      ub = ones(self.grid.ndims,dtype=int)
      for idim in range(self.grid.ndims):
        lb[idim] = coords[idim] * domain_size[idim]
        ub[idim] = (coords[idim]+1) * domain_size[idim]
        if (coords[idim] == num_domains[idim]-1):
          ub[idim] = self.grid.numCells[idim]
      self.slabs.append(Slab(lb,ub))
    
    logger.debug("Domain decomposed into slabs:")
    for slab in self.slabs:
      logger.debug("lb: "+str(slab.lowerBounds)+", ub: "+str(slab.upperBounds))

  def plot(self, axes=None, plot=False, by_index=False):
    if (self.grid.ndims == 3):
      if not axes:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_aspect("equal")
      else:
        ax = axes

      for slab in self.slabs:
        if by_index:
          PlotRectangles.draw_3D_box(ax,slab)
        else:
          PlotRectangles.draw_3D_box(ax,Slab(self.grid.get_pos_at_ind(slab.lowerBounds), \
              self.grid.get_pos_at_ind(slab.upperBounds)))
    
    elif (self.grid.ndims == 2):
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      else:
        ax = axes
      
      for slab in self.slabs:
        PlotRectangles.draw_2D_box(ax,slab)
      
    else:
      if not axes:
        fig = plt.figure()
        ax = fig.subplot(111)
        ax.set_aspect("equal")
      else:
        ax = axes
      
      for slab in self.slabs:
        PlotRectangles.draw_1D_box(ax,slab)
    
    if (plot):
      plt.show()
    
    return ax
  
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

class Geometry:
  def __init__(self,file):
    self.stl_mesh = mesh.Mesh.from_file(file)
  
  def get_vertices(self):
    return self.stl_mesh.points.reshape([-1, 3])
  
  def get_triangles(self):
    return self.stl_mesh.points

  def check_intersection(self,v0,v1,v2,h):
    # checks intersection of triangle defined by v0, v1, v2 points
    #   and cube centered at origin with half-side length h

    #######
    # first check is an axis check, 9 separate tests

    # get edges of triangles
    e0 = v1-v0 
    e1 = v2-v1
    e2 = v0-v2

    # now do the tests, only return if False, otherwise keep going
    if not self.__axis_test_x01(e0[2],e0[1],abs(e0[2]),abs(e0[1]),v0,v1,v2,h): return False
    if not self.__axis_test_y02(e0[2],e0[0],abs(e0[2]),abs(e0[0]),v0,v1,v2,h): return False
    if not self.__axis_test_z12(e0[1],e0[0],abs(e0[1]),abs(e0[0]),v0,v1,v2,h): return False

    if not self.__axis_test_x01(e1[2],e1[1],abs(e1[2]),abs(e1[1]),v0,v1,v2,h): return False
    if not self.__axis_test_y02(e1[2],e1[0],abs(e1[2]),abs(e1[0]),v0,v1,v2,h): return False
    if not self.__axis_test_z0(e1[1],e1[0],abs(e1[1]),abs(e1[0]),v0,v1,v2,h): return False

    if not self.__axis_test_x2(e2[2],e2[1],abs(e2[2]),abs(e2[1]),v0,v1,v2,h): return False
    if not self.__axis_test_y1(e2[2],e2[0],abs(e2[2]),abs(e2[0]),v0,v1,v2,h): return False
    if not self.__axis_test_z12(e2[1],e2[0],abs(e2[1]),abs(e2[0]),v0,v1,v2,h): return False

    #######
    # next we check if the bounding square of the triangle intersects the cube
    #  if any of these is not the case then it cannot intersect, return false
    if(min(array([v0[0],v1[0],v2[0]])) > h[0] or max(array([v0[0],v1[0],v2[0]])) < -h[0]):
      return False
    if(min(array([v0[1],v1[1],v2[1]])) > h[1] or max(array([v0[1],v1[1],v2[1]])) < -h[1]):
      return False
    if(min(array([v0[2],v1[2],v2[2]])) > h[2] or max(array([v0[2],v1[2],v2[2]])) < -h[2]):
      return False

    #######
    # last we check if the line defined by the cross product of a triangle 
    #   edge with each unit vector intersects the box
    normal = cross(e0,e1)
    vmin = ones(3)
    vmax = ones(3)

    for idim in range (3):
      if (normal[idim] > 0.0):
        vmin[idim] = -h[idim] - v0[idim]
        vmax[idim] =  h[idim] - v0[idim]
      else:
        vmin[idim] =  h[idim] - v0[idim]
        vmax[idim] = -h[idim] - v0[idim]

    if(dot(normal,vmin)>0.0):
      return False # err on the side of no
    if(dot(normal,vmax)>=0.0): # not a typo
      return True
    return False

  # x-tests
  def __axis_test_x01(self,a,b,fa,fb,v0,v1,v2,h):
    p0 = a*v0[1] - b*v0[2]
    p2 = a*v2[1] - b*v2[2]
    mini,maxi = [p0,p2] if p0<p2 else [p2,p0]
    rad = fa * h[1] + fb * h[2]
    return False if(mini>rad or maxi<-rad) else True

  def __axis_test_x2(self,a,b,fa,fb,v0,v1,v2,h):
    p0 = a*v0[1] - b*v0[2]
    p1 = a*v1[1] - b*v1[2]
    mini,maxi = [p0,p1] if p0<p1 else [p1,p0]
    rad = fa * h[1] + fb * h[2]
    return False if(mini>rad or maxi<-rad) else True

  # y-tests
  def __axis_test_y02(self,a,b,fa,fb,v0,v1,v2,h):
    p0 = -a*v0[0] + b*v0[2]
    p2 = -a*v2[0] + b*v2[2]
    mini,maxi = [p0,p2] if p0<p2 else [p2,p0]
    rad = fa * h[0] + fb * h[2]
    return False if(mini>rad or maxi<-rad) else True

  def __axis_test_y1(self,a,b,fa,fb,v0,v1,v2,h):
    p0 = -a*v0[0] + b*v0[2]
    p1 = -a*v1[0] + b*v1[2]
    mini,maxi = [p0,p1] if p0<p1 else [p1,p0]
    rad = fa * h[0] + fb * h[2]
    return False if(mini>rad or maxi<-rad) else True

  # z-tests
  def __axis_test_z12(self,a,b,fa,fb,v0,v1,v2,h):
    p1 = a*v1[0] - b*v1[1]
    p2 = a*v2[0] - b*v2[1]
    mini,maxi = [p1,p2] if p1<p2 else [p2,p1]
    rad = fa * h[0] + fb * h[1]
    return False if(mini>rad or maxi<-rad) else True

  def __axis_test_z0(self,a,b,fa,fb,v0,v1,v2,h):
    p0 = a*v0[0] - b*v0[1]
    p1 = a*v1[0] - b*v1[1]
    mini,maxi = [p0,p1] if p0<p1 else [p1,p0]
    rad = fa * h[1] + fb * h[2]
    return False if(mini>rad or maxi<-rad) else True

class PlotRectangles:

  def draw_3D_box(ax, slab:Slab, draw_surfaces = False):
    x_range = slab.get_range(0)
    y_range = slab.get_range(1)
    z_range = slab.get_range(2)

    xx, yy = meshgrid(x_range, y_range)
    zz0 = array([[z_range[0],z_range[0]],[z_range[0],z_range[0]]])
    zz1 = array([[z_range[1],z_range[1]],[z_range[1],z_range[1]]])
    ax.plot_wireframe(xx, yy, zz0, color="r")
    ax.plot_wireframe(xx, yy, zz1, color="r")
    if (draw_surfaces):
      ax.plot_surface(xx, yy, zz0, color="r", alpha=0.2)
      ax.plot_surface(xx, yy, zz1, color="r", alpha=0.2)

    yy, zz = meshgrid(y_range, z_range)
    xx0 = array([[x_range[0],x_range[0]],[x_range[0],x_range[0]]])
    xx1 = array([[x_range[1],x_range[1]],[x_range[1],x_range[1]]])
    ax.plot_wireframe(xx0, yy, zz, color="r")
    ax.plot_wireframe(xx1, yy, zz, color="r")
    if (draw_surfaces):
      ax.plot_surface(xx0, yy, zz, color="r", alpha=0.2)
      ax.plot_surface(xx1, yy, zz, color="r", alpha=0.2)

    yy0 = array([[y_range[0],y_range[0]],[y_range[0],y_range[0]]])
    yy1 = array([[y_range[1],y_range[1]],[y_range[1],y_range[1]]])
    ax.plot_wireframe(xx, yy0, zz, color="r")
    ax.plot_wireframe(xx, yy1, zz, color="r")
    if (draw_surfaces):
      ax.plot_surface(xx, yy0, zz, color="r", alpha=0.2)
      ax.plot_surface(xx, yy1, zz, color="r", alpha=0.2)
  
  def draw_2D_box(ax,slab):
    x_range = slab.get_range(0)
    y_range = slab.get_range(1)

    xx, yy = meshgrid(x_range, y_range)
    ax.plot(xx[0], yy[0], color="r")
    ax.plot(xx[0], yy[1], color="r")
    ax.plot(xx[1], yy[0], color="r")
    ax.plot(xx[1], yy[1], color="r")
  
  def draw_1D_box(ax,slab):
    x_range = slab.get_range(0)

    xx, yy = meshgrid(x_range, array([-1,1]))
    ax.plot(xx[0], yy[0], color="r")
    ax.plot(xx[0], yy[1], color="r")
    ax.plot(xx[1], yy[0], color="r")
    ax.plot(xx[1], yy[1], color="r")