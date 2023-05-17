from __future__ import annotations
from numpy import min, max, array, ndarray, all, any, product, meshgrid

import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger("requests")
requests_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
requests_logger.addHandler(handler)

class Slab:
    """Class that defines a slab based on lower and upper bounds. Can be any-dimensional."""

    def __init__(self, lb: ndarray, ub: ndarray):
        """_summary_

        Args:
            lb (ndarray): lower bounds of slab [x0,y0,...]
            ub (ndarray): upper bounds of slab [x1,y1,...]
        """
        self.lowerBounds = array(lb)
        self.upperBounds = array(ub)
        self.ndim = self.lowerBounds.size

    def __repr__(self):
        return (
            "Slab bounds: \n"
            + str(self.lowerBounds)
            + "\n"
            + str(self.upperBounds)
            + "\n"
        )

    def __str__(self):
        return (
            "Slab bounds: \n"
            + str(self.lowerBounds)
            + "\n"
            + str(self.upperBounds)
            + "\n"
        )

    def __eq__(self, other_slab: Slab) -> bool:
        """Check if slabs are identical

        Args:
            other_slab (Slab): Slab to compare

        Returns:
            bool: whether slabs are equal
        """
        return (all(self.lowerBounds == other_slab.lowerBounds)) and (
            all(self.upperBounds == other_slab.upperBounds)
        )

    def get_range(self, idim: int) -> ndarray:
        """Get the range of a particular dimension of the slab

        Args:
            idim (int): dimension for which the bounds are desired

        Returns:
            ndarray: upper and lower bounds of the requested dimension
        """
        return array([self.lowerBounds[idim], self.upperBounds[idim]])

    def get_lengths(self) -> ndarray:
        """Get the lengths of the slab in each dimension

        Returns:
            ndarray: array of the lengths of the slab
        """
        return self.upperBounds - self.lowerBounds

    def is_empty(self) -> bool:
        """Get flag for whether slab is empty or not. A slab is considered empty
        if any of its dimensional lengths are zero or less

        Returns:
            bool: flag for whether the slab is empty or not
        """
        return any(self.get_lengths() <= 0)

    def get_volume(self) -> float:
        """Get the volume of the slab (ie. the product of its lengths)

        Returns:
            float: volume of the slab
        """
        return product(self.get_lengths())

    def set_empty(self) -> None:
        """Set the slab to be empty"""
        self.lowerBounds = array([0, 0, 0])
        self.upperBounds = array([0, 0, 0])

    def neighbors(self, check_slab: Slab) -> bool:
        """Determine whether this slab and the provided one are neighbors (ie. do their edges touch)

        Args:
            check_slab (Slab): slab to check for neighborhood

        Returns:
            bool: whether the slabs are neighbors or not
        """
        union_length = self.union(check_slab).get_lengths()
        total_length = self.get_lengths() + check_slab.get_lengths()
        if any(union_length > total_length):
            return False
        return True

    def overlaps(self, check_slab: Slab) -> bool:
        """Check whether this slab overlaps the provided slab

        Args:
            slab (Slab): slab to check overlap with

        Returns:
            bool: whether slabs overlap or not
        """
        intersection = self.intersection(check_slab)
        if intersection.get_volume() > 0:
            return True
        return False

    def union(self, check_slab: Slab) -> Slab:
        """Get the smallest slab that contains both this slab and the provided one

        Args:
            check_slab (Slab): slab to union with

        Returns:
            Slab: minimal slab containing both this slab and provided slab
        """
        return Slab(
            min(array([self.lowerBounds, check_slab.lowerBounds]), axis=0),
            max(array([self.upperBounds, check_slab.upperBounds]), axis=0),
        )

    def intersection(self, check_slab: Slab) -> Slab:
        """Get the largest region contained by this slab and the provided slab

        Args:
            check_slab (Slab): slab to check intersection with

        Returns:
            Slab: maximum slab contained by both this and provided slab
        """
        return Slab(
            max(array([self.lowerBounds, check_slab.lowerBounds]), axis=0),
            min(array([self.upperBounds, check_slab.upperBounds]), axis=0),
        )


class IndexSlab:
    """Class that creates a slab of indices and allows for each conversion between
    linear and vector indices.
    """

    def __init__(self, nx: ndarray) -> None:
        """Initialize the IndexSlab

        Args:
            nx (ndarray): list of the lengths of each dimension
        """
        self.nx = array(nx)
        self.ndim = len(nx)

    def getIndices(self, linInd: int) -> ndarray[int]:
        """Get the vector indices for the specified linear index

        Args:
            linInd (int): linear index

        Returns:
            ndarray[int]: array of the vector indices
        """
        if self.ndim == 1:
            return array([linInd])
        elif self.ndim == 2:
            return array([linInd % self.nx[0], linInd // self.nx[0]])
        else:
            nxny = self.nx[0] * self.nx[1]
            return array(
                [linInd % self.nx[0], (linInd % nxny) // self.nx[0], linInd // nxny]
            )


class PlotRectangles:
    """Unitility class for plotting rectangles with edges and surfaces"""

    def draw_3D_box(ax, slab: Slab, draw_surfaces=False):
        """Plot 3D box on axis ax

        Args:
            ax (axis): 3D matplotlib axis on which to plot the rectangle
            slab (Slab): slab that defines the box
            draw_surfaces (bool, optional): whether or not to draw the box faces. Defaults to False.
        """
        x_range = slab.get_range(0)
        y_range = slab.get_range(1)
        z_range = slab.get_range(2)

        xx, yy = meshgrid(x_range, y_range)
        zz0 = array([[z_range[0], z_range[0]], [z_range[0], z_range[0]]])
        zz1 = array([[z_range[1], z_range[1]], [z_range[1], z_range[1]]])
        ax.plot_wireframe(xx, yy, zz0, color="r")
        ax.plot_wireframe(xx, yy, zz1, color="r")
        if draw_surfaces:
            ax.plot_surface(xx, yy, zz0, color="r", alpha=0.2)
            ax.plot_surface(xx, yy, zz1, color="r", alpha=0.2)

        yy, zz = meshgrid(y_range, z_range)
        xx0 = array([[x_range[0], x_range[0]], [x_range[0], x_range[0]]])
        xx1 = array([[x_range[1], x_range[1]], [x_range[1], x_range[1]]])
        ax.plot_wireframe(xx0, yy, zz, color="r")
        ax.plot_wireframe(xx1, yy, zz, color="r")
        if draw_surfaces:
            ax.plot_surface(xx0, yy, zz, color="r", alpha=0.2)
            ax.plot_surface(xx1, yy, zz, color="r", alpha=0.2)

        yy0 = array([[y_range[0], y_range[0]], [y_range[0], y_range[0]]])
        yy1 = array([[y_range[1], y_range[1]], [y_range[1], y_range[1]]])
        ax.plot_wireframe(xx, yy0, zz, color="r")
        ax.plot_wireframe(xx, yy1, zz, color="r")
        if draw_surfaces:
            ax.plot_surface(xx, yy0, zz, color="r", alpha=0.2)
            ax.plot_surface(xx, yy1, zz, color="r", alpha=0.2)

    def draw_2D_box(ax, slab: Slab):
        """Plot 2D box on axis ax

        Args:
            ax (axis): 2D matplotlib axis on which to plot the rectangle
            slab (Slab): slab that defines the box
        """
        x_range = slab.get_range(0)
        y_range = slab.get_range(1)

        xx, yy = meshgrid(x_range, y_range)
        ax.plot(xx[0], yy[0], color="r")
        ax.plot(xx[0], yy[1], color="r")
        ax.plot(xx[1], yy[0], color="r")
        ax.plot(xx[1], yy[1], color="r")

    def draw_1D_box(ax, slab: Slab):
        """Plot 1D box on axis ax. Box will appear as vertical, red lines on the axis

        Args:
            ax (axis): 2D matplotlib axis on which to plot the rectangle
            slab (Slab): slab that defines the box
        """
        x_range = slab.get_range(0)

        xx, yy = meshgrid(x_range, array([-1, 1]))
        ax.plot(xx[0], yy[0], color="r")
        ax.plot(xx[0], yy[1], color="r")
        ax.plot(xx[1], yy[0], color="r")
        ax.plot(xx[1], yy[1], color="r")
