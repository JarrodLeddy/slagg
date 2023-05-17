from numpy import ndarray, abs, max, array, cross, ones, dot

from stl import mesh

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger("requests")
requests_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
requests_logger.addHandler(handler)


class Geometry:
    """Class for storing geometry data from stl"""

    def __init__(self, file: str):
        """Initialize Geometry object

        Args:
            file (str): STL file location and name
        """
        self.stl_mesh = mesh.Mesh.from_file(file)

    def get_vertices(self) -> ndarray:
        """Get the verticies of the triangles

        Returns:
            ndarray: triangle vertices
        """
        return self.stl_mesh.points.reshape([-1, 3])

    def get_triangles(self) -> ndarray:
        """Get the triangles in the form [N,9] where axis 1 holds the nine
        coordinates (x,y,z) of the 3 triangle vertices

        Returns:
            ndarray: triangle vertices
        """
        return self.stl_mesh.points

    def plot(self, plot=False):
        """Plot the Goemetry object

        Args:
            plot (bool, optional): Whether to show the plot at the end. Defaults to False.

        Returns:
            axis: 3D axis to which more plots can be added
        """

        # Create a new plot
        figure = plt.figure()
        ax = Axes3D(figure, auto_add_to_figure=False)
        figure.add_axes(ax)
        ax.add_collection3d(Poly3DCollection(self.stl_mesh.vectors))
        scale = self.stl_mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        if plot:
            plt.show()
        else:
            return ax

    def check_tricube_intersection(
        self, v0: ndarray, v1: ndarray, v2: ndarray, h: ndarray
    ) -> bool:
        """Checks a single triange (with vertices v0,v1,v2) for intersection with a cube
        centered at (0,0,0) with side length 2*h

        Args:
            v0 (ndarray): coordinates of first triangle vertex
            v1 (ndarray): coordinates of second triangle vertex
            v2 (ndarray): coordinates of third triangle vertex
            h (ndarray): half-length of box edge

        Returns:
            bool: whether the cube and triangle intercept
        """
        # checks intersection of triangle defined by v0, v1, v2 points
        #   and cube centered at origin with half-side length h

        # get edges of triangles
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        #######
        # first check is an axis check, 9 separate tests
        if not self.__axis_test_x01(
            e0[2], e0[1], abs(e0[2]), abs(e0[1]), v0, v1, v2, h
        ):
            return False
        if not self.__axis_test_y02(
            e0[2], e0[0], abs(e0[2]), abs(e0[0]), v0, v1, v2, h
        ):
            return False
        if not self.__axis_test_z12(
            e0[1], e0[0], abs(e0[1]), abs(e0[0]), v0, v1, v2, h
        ):
            return False

        if not self.__axis_test_x01(
            e1[2], e1[1], abs(e1[2]), abs(e1[1]), v0, v1, v2, h
        ):
            return False
        if not self.__axis_test_y02(
            e1[2], e1[0], abs(e1[2]), abs(e1[0]), v0, v1, v2, h
        ):
            return False
        if not self.__axis_test_z0(e1[1], e1[0], abs(e1[1]), abs(e1[0]), v0, v1, v2, h):
            return False

        if not self.__axis_test_x2(e2[2], e2[1], abs(e2[2]), abs(e2[1]), v0, v1, v2, h):
            return False
        if not self.__axis_test_y1(e2[2], e2[0], abs(e2[2]), abs(e2[0]), v0, v1, v2, h):
            return False
        if not self.__axis_test_z12(
            e2[1], e2[0], abs(e2[1]), abs(e2[0]), v0, v1, v2, h
        ):
            return False

        #######
        # next we check if the bounding square of the triangle intersects the cube
        #  if any of these is not the case then it cannot intersect, return false
        if (
            min(array([v0[0], v1[0], v2[0]])) > h[0]
            or max(array([v0[0], v1[0], v2[0]])) < -h[0]
        ):
            return False
        if (
            min(array([v0[1], v1[1], v2[1]])) > h[1]
            or max(array([v0[1], v1[1], v2[1]])) < -h[1]
        ):
            return False
        if (
            min(array([v0[2], v1[2], v2[2]])) > h[2]
            or max(array([v0[2], v1[2], v2[2]])) < -h[2]
        ):
            return False

        #######
        # last we check if the line defined by the cross product of a triangle
        #   edge with each unit vector intersects the box
        normal = cross(e0, e1)
        vmin = ones(3)
        vmax = ones(3)

        for idim in range(3):
            sign = 1.0 if (normal[idim] > 0.0) else -1.0
            vmin[idim] = -sign * h[idim] - v0[idim]
            vmax[idim] = sign * h[idim] - v0[idim]

        if dot(normal, vmin) > 0.0:
            return False  # err on the side of false
        if dot(normal, vmax) >= 0.0:  # not a typo
            return True
        return False

    # x-tests
    def __axis_test_x01(self, a, b, fa, fb, v0, v1, v2, h):
        p0 = a * v0[1] - b * v0[2]
        p2 = a * v2[1] - b * v2[2]
        mini, maxi = [p0, p2] if p0 < p2 else [p2, p0]
        rad = fa * h[1] + fb * h[2]
        return False if (mini > rad or maxi < -rad) else True

    def __axis_test_x2(self, a, b, fa, fb, v0, v1, v2, h):
        p0 = a * v0[1] - b * v0[2]
        p1 = a * v1[1] - b * v1[2]
        mini, maxi = [p0, p1] if p0 < p1 else [p1, p0]
        rad = fa * h[1] + fb * h[2]
        return False if (mini > rad or maxi < -rad) else True

    # y-tests
    def __axis_test_y02(self, a, b, fa, fb, v0, v1, v2, h):
        p0 = -a * v0[0] + b * v0[2]
        p2 = -a * v2[0] + b * v2[2]
        mini, maxi = [p0, p2] if p0 < p2 else [p2, p0]
        rad = fa * h[0] + fb * h[2]
        return False if (mini > rad or maxi < -rad) else True

    def __axis_test_y1(self, a, b, fa, fb, v0, v1, v2, h):
        p0 = -a * v0[0] + b * v0[2]
        p1 = -a * v1[0] + b * v1[2]
        mini, maxi = [p0, p1] if p0 < p1 else [p1, p0]
        rad = fa * h[0] + fb * h[2]
        return False if (mini > rad or maxi < -rad) else True

    # z-tests
    def __axis_test_z12(self, a, b, fa, fb, v0, v1, v2, h):
        p1 = a * v1[0] - b * v1[1]
        p2 = a * v2[0] - b * v2[1]
        mini, maxi = [p1, p2] if p1 < p2 else [p2, p1]
        rad = fa * h[0] + fb * h[1]
        return False if (mini > rad or maxi < -rad) else True

    def __axis_test_z0(self, a, b, fa, fb, v0, v1, v2, h):
        p0 = a * v0[0] - b * v0[1]
        p1 = a * v1[0] - b * v1[1]
        mini, maxi = [p0, p1] if p0 < p1 else [p1, p0]
        rad = fa * h[1] + fb * h[2]
        return False if (mini > rad or maxi < -rad) else True
