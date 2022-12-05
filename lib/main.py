from OCC.Core.BRepPrimAPI import *
from OCC.Display.WebGl import x3dom_renderer
from OCC.Core.gp import *

from core import *

import sys

mysolve = None
mygrid = None
mygeo = None

def main():
  filename = sys.argv[1]
  scale = sys.argv[2]
  dimx = sys.argv[3]
  dimy = sys.argv[4]
  dimz = sys.argv[5]
  numrects = sys.argv[6]

  perform_setup(filename, int(dimx), int(dimy), int(dimz), scale, int(numrects))  

def draw_trial_geo(res, dimx, dimy, dimz, scale):
  my_renderer = x3dom_renderer.X3DomRenderer()

  full_x = float(scale * dimx)
  full_y = float(scale * dimy)
  full_z = float(scale * dimz)

  # geo_box = BRepPrimAPI_MakeBox(full_x, full_y, full_z)
  # my_renderer.DisplayShape(geo_box.Shape())
  
  boxes = []
  outlines = []

  for rect in res:
    lower_corner = rect.get_start_cell().real_bounds_low()
    lengths = []
    for idx in [0, 1, 2]:
      if (float(rect.length(idx) * scale)) <= 0.0:
        lengths.append(1.0)
      else:
        lengths.append(float(rect.length(idx)))

    start_pt = gp_Pnt(float(lower_corner[0]), float(lower_corner[1]), float(lower_corner[2]))
    decomp_box = BRepPrimAPI_MakeBox(start_pt, float(lengths[0] * scale), float(lengths[1] * scale), float(lengths[2] * scale)).Shape()
    # outline_box = BRepPrimAPI_MakeBox(start_pt, float((rect.length(0) + 0.01) * scale), float((rect.length(1) + 0.01) * scale), float(rect.length(2) * scale)).Shape()
    boxes.append(decomp_box)
    # outlines.append(outline_box)

  for b in boxes:
    my_renderer.DisplayShape(b, color = [0, 0, 255])
  # for o in outlines:
  #   my_renderer.DisplayShape(o, color = [0, 0, 0])

  my_renderer.render()
  
def perform_setup(filename, dimx, dimy, dimz, scale, numrects):
  mygeo = geometry.Geometry(filename)
  shape = mygeo.read_cad()
  print("Read geometry file " + filename)
  
  mygrid = grid.Grid(scale, [dimx, dimy, dimz], mygeo)
  # mygeo.visualize()
  
  mysolve = solver.Solver(mygrid, numrects)
  res = mysolve.run_solver()
  print(res)
  print(len(res))
  draw_trial_geo(res, int(dimx), int(dimy), int(dimz), float(scale))
  
if __name__ == '__main__':
  main()