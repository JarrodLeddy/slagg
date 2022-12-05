import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from stl import mesh
import sys

from OCC.Core.BRepPrimAPI import *
from OCC.Display.WebGl import x3dom_renderer
from OCC.Core.gp import *
from OCC.Core.BRepTools import *
from OCC.Extend.TopologyUtils import *
from OCC.Core.StlAPI import StlAPI_Writer

from core import *

mysolve = None
mygrid = None
mygeo = None

def main():
  filename = sys.argv[1]
  dimx = sys.argv[2]
  dimy = sys.argv[3]
  dimz = sys.argv[4]
  scale = sys.argv[5]
  numrects = sys.argv[6]

  perform_setup(filename, int(dimx), int(dimy), int(dimz), float(scale), int(numrects))  

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

def dev_draw(res, dimx, dimy, dimz, scale, shape, nudge):
  full_x = float(scale * dimx)
  full_y = float(scale * dimy)
  full_z = float(scale * dimz)
  
  # Write the geometry to an stl for use with numpy-stl
  # stl_writer = StlAPI_Writer()
  # stl_writer.SetASCIIMode(True)
  # stl_writer.Write(shape, 'shape.stl')
  
  fig = plt.figure()
  ax = mplot3d.Axes3D(fig)
  
  my_mesh = mesh.Mesh.from_file('shape.stl')
  vecs = mplot3d.art3d.Poly3DCollection(my_mesh.vectors)
  vecs.set_alpha(0.25)
  ax.add_collection3d(vecs)
  
  xs = np.zeros(shape=(1, len(res)))
  ys = np.zeros(shape=(1, len(res)))
  zs = np.zeros(shape=(1, len(res)))
  
  for i in range(len(res)):
    lower_corner = res[i].get_start_cell().real_bounds_low()
    xs[0][i] = float(lower_corner[0] - nudge[0])
    ys[0][i] = float(lower_corner[1] - nudge[1])
    zs[0][i] = float(lower_corner[2] - nudge[2])
    
  ax.scatter(xs, ys, zs, marker='o', color='red')
  
  scale = my_mesh.points.flatten("C")
  ax.auto_scale_xyz(scale, scale, scale)
  
  plt.show()
  
def dev_draw_decomp(res, dimx, dimy, dimz, scale, shape):
  fig = plt.figure()
  ax = mplot3d.Axes3D(fig)
  
  my_mesh = mesh.Mesh.from_file('shape.stl')
  vecs = mplot3d.art3d.Poly3DCollection(my_mesh.vectors)
  vecs.set_alpha(0.25)
  ax.add_collection3d(vecs)
  
  for r in res:
    r.draw(ax)
    
  plt.show()

def perform_setup(filename, dimx, dimy, dimz, scale, numrects):
  mygeo = geometry.Geometry(filename)
  shape = mygeo.read_cad()
  topo = TopologyExplorer(shape)
  print("Read geometry file " + filename)
  verts = topo.vertices()
  
  nudge = [0.0, 0.0, 0.0]
  
  brt = BRep_Tool()
  for v in verts:
    pnt = brt.Pnt(v)
    if abs(pnt.X()) > nudge[0] and pnt.X() < 0.0:
      nudge[0] = abs(pnt.X())
    if abs(pnt.Y()) > nudge[1] and pnt.Y() < 0.0:
      nudge[1] = abs(pnt.Y())
    if abs(pnt.Z()) > nudge[2] and pnt.Z() < 0.0:
      nudge[2] = abs(pnt.Z())      
  
  mygrid = grid.Grid(scale, [dimx, dimy, dimz], shape, nudge)
  # mygeo.visualize()
  
  mysolve = solver.Solver(mygrid, numrects)
  res = mysolve.run_solver()
  # draw_trial_geo(res, int(dimx), int(dimy), int(dimz), float(scale))
  # dev_draw(res, int(dimx), int(dimy), int(dimz), float(scale), shape, nudge)
  dev_draw_decomp(res, int(dimx), int(dimy), int(dimz), float(scale), shape)
  
if __name__ == '__main__':
  main()