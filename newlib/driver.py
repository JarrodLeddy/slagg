import stl
from grid import Grid, Decomp, Geometry

def runTest():
  dim = 3
  nx = [30,30,30]
  sp = [0.0,0.0,0.0]
  ep = [1.0,1.0,1.0]

  # create geometry
  geom = Geometry('Moon.stl')

  # create grid
  # grid = Grid((10,15,20),(-1.0,-1.0,-1.0),(1.0,2.0,1.0))
  grid = Grid((17,4,30), geometry=geom)
  ax = grid.plot(plot=False)

  decomp = Decomp(grid,12)
  _ = decomp.plot(axes=ax, plot=True)

if __name__ == "__main__":
  runTest()