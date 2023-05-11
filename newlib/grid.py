from numpy import array

class Slab:
  lowerBounds = ()
  upperBounds = ()

  def __init__(self, lb:tuple, ub:tuple):
    self.lowerBounds = lb
    self.upperBounds = ub

class Cell:
  hasGeom = False

  def __init__(self, inds, pos):
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
    self.numCells = numCells
    self.ndims = len(numCells)
    print(numCells,len(numCells))
    self.dx = tuple((array(endPos)-array(startPos)) / array(numCells))
    self.slab = Slab(tuple(0 for i in self.numCells),numCells)
    self.posSlab = Slab(startPos,endPos)

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