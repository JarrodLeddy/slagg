import stl
from grid import Grid, Decomp

def runTest():
  dim = 3
  nx = [30,30,30]
  sp = [0.0,0.0,0.0]
  ep = [1.0,1.0,1.0]

  # create geometry

  # create grid
  grid = Grid((10,15,20),(-1.0,-1.0,-1.0),(1.0,2.0,1.0))
  decomp = Decomp(grid,9)
  decomp.plot()

if __name__ == "__main__":
  runTest()