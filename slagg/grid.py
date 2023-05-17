from __future__ import annotations
from slagg.utils import Slab, IndexSlab, PlotRectangles
from slagg.geometry import Geometry

from numpy import (
    array,
    ndarray,
    argmax,
    copy,
    ones,
    min,
    max,
    floor,
    mgrid,
    dot,
    zeros,
    sum,
)
import matplotlib.pyplot as plt

import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

requests_logger = logging.getLogger("requests")
requests_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
requests_logger.addHandler(handler)


class Cell:
    """Class for storing a cell (box) for simulation. Holds a slab of indices, a position
    corresponding to the bottom left corner of the box, the edge length of the box
    (assumed cubic), and a flag for whether the cell has any geometry in it.
    """

    has_geometry = False

    def __init__(
        self,
        inds: ndarray[int],
        pos: ndarray[float],
        dx: float,
        contains_geometry=False,
    ):
        """Initialize a Cell object

        Args:
            inds (ndarray[int]): vector index for the box location
            pos (ndarray[float]): vector of physical position of the box location (lower left corner)
            dx (float): edge length of the box
            contains_geometry (bool, optional): whether the box is inside some geometry. Defaults to False.
        """
        self.position = array(pos)
        self.indices = array(inds)
        self.slab = Slab(self.indices, self.indices + 1)
        self.dx = dx
        self.has_geometry = contains_geometry

    def set_has_geometry(self, hgb: bool) -> None:
        """Sets the internal flag for the cell for whether or not it is inside
        a geometry object

        Args:
            hgb (bool): state to set. True it is inside geometry; False
            it is outside geometry
        """
        self.has_geometry = hgb

    def get_center(self) -> ndarray[float]:
        """Get the location of the center of the cell (cell.position is the location
        of the lower left corner)

        Returns:
            ndarray[float]: The position of the center of the cell
        """
        return self.position + 0.5 * self.dx


class Grid:
    """Class with data and methods for storing and computing grid quantities"""

    numCells = ()
    ndims = ()
    dx = ()
    slab = None
    posSlab = None
    cells = dict()
    geometry = None

    def __init__(
        self,
        numCells: tuple,
        startPos: tuple = None,
        endPos: tuple = None,
        geometry: Geometry = None,
    ):
        """Initialize the Grid object. If startPos and endPos are not provided, the
        geometry must be and will be used to calculate startPos and endPos automatically.
        If numCells contains only one number with None other arguments (ie. numCells
        = [50,None,None]), then the largest dimension of the geometry will be given the
        requested number of cells and the other dimensions will be scaled accordingly to
        form cubic cells.

        Args:
            numCells (tuple): number of cells in each dimension
            startPos (tuple, optional): start position of the grid. Defaults to None.
            endPos (tuple, optional): end position of the grid. Defaults to None.
            geometry (Geometry, optional): geometry with which to calculate intersections with the grid. Defaults to None.
        """

        self.ndims = len(numCells)

        # check if geometry defined, if not then endPos and startPos must be
        if (startPos is None or endPos is None) and (geometry is None):
            raise (
                Exception(
                    "SLAGG Grid error: Either geometry must be specified or start/end positions"
                )
            )
        elif geometry is not None:
            self.geometry = geometry
            verts = geometry.get_vertices()
            sp = ones(self.ndims)
            ep = ones(self.ndims)

            # find min and max of geometry, set startPos and endPos there
            for i in range(self.ndims):
                sp[i] = min(verts[:, i])
                ep[i] = max(verts[:, i])

            # debug output so the user can see the geometry loaded correctly
            logger.debug("Found geometry bounds:")
            logger.debug("start positions:  " + str(sp))
            logger.debug("end positions:    " + str(ep) + "\n")

            # now shift, add normalized padding, and shift back
            osp = copy(sp)
            lengths = array(ep) - array(sp)
            ep -= sp
            sp -= sp
            startPos = (sp - 0.05 * lengths) + osp
            endPos = (ep + 0.05 * lengths) + osp

            # tell the user what grid bounds were chosen
            logger.info("Using geometry to determine the grid size:")
            logger.info("start positions:  " + str(startPos))
            logger.info("end positions:    " + str(endPos) + "\n")

        if numCells[-1] == None:
            maxCells = numCells[0]
            numCells = zeros(3, dtype=int)
            maxind = argmax(lengths)
            numCells[maxind] = maxCells
            for i in range(3):
                if i != maxind:
                    numCells[i] = int(maxCells * lengths[i] / lengths[maxind])

        else:
            if (
                len(numCells) != len(startPos)
                or len(numCells) != len(endPos)
                or len(startPos) != len(endPos)
            ):
                raise (
                    Exception(
                        "SLAGG Grid error: specified grid dimensionality not consistent in startPos, endPos, and numCells"
                    )
                )

        self.numCells = array(numCells)
        self.dx = (array(endPos) - array(startPos)) / array(self.numCells)
        self.slab = Slab(array([0 for i in self.numCells]), array(self.numCells))
        self.posSlab = Slab(array(startPos), array(endPos))
        self.lengths = array(endPos) - array(startPos)
        logger.info("Initializing Grid with " + str(self.numCells) + " cells")

        # generate set of cells
        if self.ndims == 1:
            for i in range(self.numCells[0]):
                self.cells[(i)] = Cell((i,), (startPos[0] + i * self.dx[0],), self.dx)
        elif self.ndims == 2:
            for i in range(self.numCells[0]):
                for j in range(self.numCells[1]):
                    self.cells[(i, j)] = Cell(
                        (i, j),
                        (startPos[0] + i * self.dx[0], startPos[1] + j * self.dx[1]),
                        self.dx,
                    )
        elif self.ndims == 3:
            for i in range(self.numCells[0]):
                for j in range(self.numCells[1]):
                    for k in range(self.numCells[2]):
                        self.cells[(i, j, k)] = Cell(
                            (i, j, k),
                            (
                                startPos[0] + i * self.dx[0],
                                startPos[1] + j * self.dx[1],
                                startPos[2] + k * self.dx[2],
                            ),
                            self.dx,
                        )
        else:
            raise (Exception("SLAGG error: grids must be 1, 2, or 3-dimensional."))

        # set geometry flag for every cell that contains a vertex
        if self.geometry is not None:
            self.__check_geometry_intersections()
            # self.__fill_between_intersections()

    def get_cell(self, inds: tuple) -> Cell:
        """Return cell at a given vector index

        Args:
            inds (tuple): vector index

        Returns:
            Cell: cell at the provided vector index
        """
        return self.cells[tuple(inds)]

    def get_ind_at_pos(self, pos: ndarray, round=False) -> ndarray:
        """Get array of indices for a given set of physical coordinates

        Args:
            pos (ndarray): set of physical coordinates
            round (bool, optional): Whether the indices should be rounded or floats. Defaults to False.

        Returns:
            ndarray: set of vector indices corresponding to the provided physical coordinates
        """
        if not round:
            return (array(pos) - self.posSlab.lowerBounds) / array(
                self.lengths
            ) * self.numCells + self.slab.lowerBounds
        return (
            array(
                floor(
                    (array(pos) - self.posSlab.lowerBounds)
                    / array(self.lengths)
                    * self.numCells
                ),
                dtype=int,
            )
            + self.slab.lowerBounds
        )

    def get_pos_at_ind(self, ind: ndarray) -> ndarray:
        """Get array of positions for a given vector index

        Args:
            ind (ndarray): vector index

        Returns:
            ndarray: array of physical coordinates corresponding to ind
        """
        return (array(ind) - array(self.slab.lowerBounds)) / array(
            self.numCells
        ) * self.lengths + self.posSlab.lowerBounds

    def set_geometry(self, geometry: Geometry) -> None:
        """Add geometry to the grid. Checks for intersections between grid
        cells and the geometry triangles. Marks cells that are intersected
        as having geometry.  Then marks cells between these intersected ones
        to also be inside the geometry.  NOTE: Grid only supports a single geometry.

        Args:
            geometry (Geometry): geometry to check for intersections with grid.
        """
        self.geoemtry = geometry
        self.__check_geometry_intersections()
        self.__fill_between_intersections()
        return

    def __check_geometry_intersections(self):
        """For each cell, check (up-to) all triangles for an intersection.
        Algorithm based on Fast 3D Triangle-Box Overlap Testing by Tomas Akenine-Moller.
        """
        logger.info(
            "Checking "
            + str(self.geometry.get_triangles().shape[0])
            + " triangles in geometry for intersection with "
            + str(len(self.cells.values()))
            + " grid cells.\n"
        )
        for c in self.cells.values():
            shift = c.get_center()
            for t in self.geometry.get_triangles():
                t0, t1, t2 = [t[0:3], t[3:6], t[6:9]]
                # shift everything so that cube is centered on (0,0,0)
                p0 = t0 - shift
                p1 = t1 - shift
                p2 = t2 - shift

                if self.geometry.check_tricube_intersection(p0, p1, p2, self.dx / 2):
                    c.set_has_geometry(True)
                    break

    def __fill_between_intersections(self) -> None:
        """Mark cells between geometry intersections as also being inside the geometry.
        This method works on two assumptions: (1) geometries are more than one cell wide
        and (2) the edge of the simulation domain is *outside* the geometry, not inside.
        """
        # assuming that no geometry is only one cell thick, so we want to fill
        #   has_geometry flag with True for all cells bewteen other trues
        for i in range(self.numCells[0]):
            for j in range(self.numCells[1]):
                inside = False
                for k in range(self.numCells[2]):
                    if self.cells[(i, j, k)].has_geometry:
                        inside = not inside
                    elif not self.cells[(i, j, k)].has_geometry and inside:
                        self.cells[(i, j, k)].set_has_geometry(True)

        for j in range(self.numCells[1]):
            for k in range(self.numCells[2]):
                inside = False
                for i in range(self.numCells[0]):
                    if self.cells[(i, j, k)].has_geometry:
                        inside = not inside
                    elif not self.cells[(i, j, k)].has_geometry and inside:
                        self.cells[(i, j, k)].set_has_geometry(True)

        for k in range(self.numCells[2]):
            for i in range(self.numCells[0]):
                inside = False
                for j in range(self.numCells[1]):
                    if self.cells[(i, j, k)].has_geometry:
                        inside = not inside
                    elif not self.cells[(i, j, k)].has_geometry and inside:
                        self.cells[(i, j, k)].set_has_geometry(True)

    def plot(self, axes=None, plot=True, rectangles=False, geometry_only=True):
        """Plot the grid object

        Args:
            axes (axis, optional): Axes on which to plot. Defaults to None.
            plot (bool, optional): Whether to show plot at the end. Defaults to False.
            rectangles (bool, optional): Whether to draw rectangles (or dots). Defaults to False.
            geometry_only (bool, optional): Whether to only plot cells that contain geometry. Defaults to True.

        Returns:
            _type_: _description_
        """
        if self.ndims == 3:
            if not axes:
                ax = plt.figure().add_subplot(projection="3d")
                #ax.set_aspect("equal")
            else:
                ax = axes

            for cell in self.cells.values():
                if (cell.has_geometry and geometry_only) or (not geometry_only):
                    if rectangles:
                        PlotRectangles.draw_3D_box(ax, cell.slab)
                    else:
                        ax.scatter(
                            cell.position[0],
                            cell.position[1],
                            cell.position[2],
                            marker=".",
                            c="k",
                        )

            # equal aspect not gauranteed in 3D, make bounding box to plot
            max_range = self.lengths.max()
            Xb = (
                0.5 * max_range * mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
                + 0.5 * self.posSlab.get_range(0).sum()
            )
            Yb = (
                0.5 * max_range * mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
                + 0.5 * self.posSlab.get_range(1).sum()
            )
            Zb = (
                0.5 * max_range * mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()
                + 0.5 * self.posSlab.get_range(2).sum()
            )
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], "w")

        elif self.ndims == 2:
            if not axes:
                fig = plt.figure()
                ax = fig.subplot(111)
                ax.set_aspect("equal")
            else:
                ax = axes

            for cell in self.cells.values():
                if (cell.has_geometry and geometry_only) or (not geometry_only):
                    if rectangles:
                        PlotRectangles.draw_2D_box(ax, cell.slab)
                    else:
                        ax.scatter(
                            cell.position[0], cell.position[1], marker=".", c="k"
                        )

        else:
            if not axes:
                fig = plt.figure()
                ax = fig.subplot(111)
                ax.set_aspect("equal")
            else:
                ax = axes

            for cell in self.cells.values():
                if (cell.has_geometry and geometry_only) or (not geometry_only):
                    if rectangles:
                        PlotRectangles.draw_1D_box(ax, cell.slab)
                    else:
                        ax.scatter(cell.position[0], marker=".", c="k")

        if plot:
            plt.show()

        return ax
