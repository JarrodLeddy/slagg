import stl
from grid import Grid, Decomp, Geometry

def runTest():
  dim = 3
  nx = [30,30,30]
  sp = [0.0,0.0,0.0]
  ep = [1.0,1.0,1.0]

  # create geometry
  geom_name = "Torus" # Moon
  geom = Geometry(geom_name+'.stl')
  #geom.plot()

  # create grid
  # grid = Grid((10,15,20),(-1.0,-1.0,-1.0),(1.0,2.0,1.0))
  if (geom_name == "Moon"):
    grid = Grid((17,4,30), geometry=geom) # with moon
  elif (geom_name == "Torus"):
    grid = Grid((20,20,7), geometry=geom)
  ax = grid.plot(plot=False)

  decomp = Decomp(grid,12)
  decomp.plot(axes=ax, plot=True)

  # refine and plot
  decomp.refine_empty()
  ax = geom.plot(plot=False)
  decomp.plot(axes=ax, plot=True)

if __name__ == "__main__":
  runTest()