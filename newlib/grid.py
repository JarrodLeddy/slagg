from numpy import (
    array,
    ndarray,
    argmax,
    copy,
    ones,
    meshgrid,
    min,
    max,
    floor,
    ceil,
    append,
    mgrid,
    cross,
    dot,
    zeros,
    sum,
    flip,
    any,
    product,
    cumsum,
)
import matplotlib.pyplot as plt
from stl import mesh
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

    def __repr__(self):
        return "Slab()"

    def __str__(self):
        return "Slab bounds: \n" + str(self.lowerBounds) + "\n" + str(self.upperBounds)

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
    numCells = ()
    ndims = ()
    dx = ()
    slab = None
    posSlab = None
    cells = dict()
    geometry = None

    def __init__(self, numCells: tuple, startPos=None, endPos=None, geometry=None):

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

    def get_cell(self, inds: tuple):
        return self.cells[tuple(inds)]

    def get_ind_at_pos(self, pos, round=False):
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

    def get_pos_at_ind(self, ind):
        return (array(ind) - array(self.slab.lowerBounds)) / array(
            self.numCells
        ) * self.lengths + self.posSlab.lowerBounds

    def set_geometry(self, geometry):
        self.geoemtry = geometry
        self.__check_geometry_intersections()
        self.__fill_between_intersections()
        return

    def __check_geometry_intersections(self):
        # Algorithm based on Fast 3D Triangle-Box Overlap Testing by Tomas Akenine-Moller
        # For each triangle, check all cells for an intersection
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

    def __fill_between_intersections(self):
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

    def plot(self, axes=None, plot=False, rectangles=False, geometry_only=True):
        if self.ndims == 3:
            if not axes:
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_aspect("equal")
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


class Decomp:
    slabs = []
    nslabs = 1

    def __init__(self, grid, nslabs):
        self.nslabs = nslabs
        self.grid = grid

        # do regular decomposition
        if (self.grid.geometry is None):
            self.__perform_regular_decomp()
        else:
            self.__perform_geometry_biased_decomp()

    def refine_empty(self, refill_empty=True):
        """Refines the decomp by removing cells empty of geometry.
        Optionally will generate more slabs if any slabs are reduced
        to zero volume

        Args:
            refill_empty (bool, optional): Option to generate more slabs
            if some were reduced to zero volume as to keep the total
            number of slabs before and after refinement the same.
            Defaults to True.
        """
        self.__squeeze_empty()

        if refill_empty:
            # if num slabs is less than desired, split largest slabs until
            #   we have the right number again
            while len(self.slabs) < self.nslabs:
                slab_vols = [slab.get_volume() for slab in self.slabs]
                logger.debug("Splitting slab: " + str(self.slabs[argmax(slab_vols)]))
                s1, s2 = self.__split_slab(self.slabs[argmax(slab_vols)])
                logger.debug("Slab split into two: " + str(s1) + "  " + str(s2))
                self.slabs[argmax(slab_vols)] = s1
                self.slabs.append(s2)

            logger.info(
                "Largest slabs split to create "
                + str(len(self.slabs))
                + " total slabs."
            )

            self.__squeeze_empty()

    def __squeeze_empty(self):
        # remove cells from decomp that have no geometry in them (assuming full row/column)
        for slab in self.slabs:
            num_cells = slab.get_lengths()
            # go through each dimension, get distributions of number of cells with geom
            has_geometry_slab = zeros(num_cells, dtype=float)
            for i in range(num_cells[0]):
                for j in range(num_cells[1]):
                    for k in range(num_cells[2]):
                        if self.grid.cells[
                            (
                                i + slab.lowerBounds[0],
                                j + slab.lowerBounds[1],
                                k + slab.lowerBounds[2],
                            )
                        ].has_geometry:
                            has_geometry_slab[i, j, k] = 1.0

            xdist = sum(has_geometry_slab, axis=(1, 2))
            ydist = sum(has_geometry_slab, axis=(0, 2))
            zdist = sum(has_geometry_slab, axis=(0, 1))

            if sum(xdist) == 0:
                slab.set_empty()
            else:
                # shorten sla bs from left
                for i, x in enumerate(xdist):
                    if x == 0:
                        slab.lowerBounds[0] += 1
                    else:
                        break

                for i, y in enumerate(ydist):
                    if y == 0:
                        slab.lowerBounds[1] += 1
                    else:
                        break

                for i, z in enumerate(zdist):
                    if z == 0:
                        slab.lowerBounds[2] += 1
                    else:
                        break

                # shorten slabs from right
                for i, x in enumerate(flip(xdist)):
                    if x == 0:
                        slab.upperBounds[0] -= 1
                    else:
                        break

                for i, y in enumerate(flip(ydist)):
                    if y == 0:
                        slab.upperBounds[1] -= 1
                    else:
                        break

                for i, z in enumerate(flip(zdist)):
                    if z == 0:
                        slab.upperBounds[2] -= 1
                    else:
                        break

        # remove any empty slabs
        self.slabs = [slab for slab in self.slabs if not slab.is_empty()]

        logger.info(
            "After initial refinement, " + str(len(self.slabs)) + " slabs remaining."
        )

    def __split_slab(self, slab: Slab) -> list:
        """Split single slab into two, biasing the split so that each
            new slab has roughly the same number of cells with geometry
            in it. The two new slabs combined cover the exact same region as
            the initial slab.

        Args:
            slab (Slab): slab to be split

        Returns:
            [Slab,Slab]: list of two new slabs that cover the same region as
            the input slab
        """
        # split slab so that each new slab has the same number of cells with geometry in it
        #   split along largest direction
        idim = argmax(slab.get_lengths())
        num_cells = slab.get_lengths()
        # go through each dimension, get distributions of number of cells with geom
        has_geometry_slab = zeros(num_cells, dtype=float)
        for i in range(num_cells[0]):
            for j in range(num_cells[1]):
                for k in range(num_cells[2]):
                    if self.grid.cells[
                        (
                            i + slab.lowerBounds[0],
                            j + slab.lowerBounds[1],
                            k + slab.lowerBounds[2],
                        )
                    ].has_geometry:
                        has_geometry_slab[i, j, k] = 1.0

        idim_dist_cum = cumsum(
            sum(has_geometry_slab, axis=((idim + 1) % 3, (idim + 2) % 3))
        )
        idim_dist_cum /= max(idim_dist_cum)

        max_loc = argmax(idim_dist_cum > 0.5)
        if max_loc == slab.get_lengths()[idim]:
            split_loc = slab.upperBounds[idim] - 1
        elif max_loc == 0:
            split_loc = slab.lowerBounds[idim] + 1
        else:
            split_loc = max_loc + slab.lowerBounds[idim]

        new_upper_bounds = copy(slab.upperBounds)
        new_upper_bounds[idim] = split_loc
        new_lower_bounds = copy(slab.lowerBounds)
        new_lower_bounds[idim] = split_loc

        return [
            Slab(slab.lowerBounds, new_upper_bounds),
            Slab(new_lower_bounds, slab.upperBounds),
        ]

    def __perform_regular_decomp(self):
        self.slabs = []
        factors = self.__prime_factors(self.nslabs)
        logger.debug(
            str(self.nslabs) + " slabs broken into prime factors: " + str(factors)
        )

        domain_size = copy(self.grid.numCells)
        num_domains = array([1 for i in self.grid.numCells])
        for f in factors:
            ind = argmax(domain_size)
            domain_size[ind] /= f
            num_domains[ind] *= f

        logger.debug("domain_size = " + str(domain_size))
        logger.debug("num_domains = " + str(num_domains))

        self.coord_map = IndexSlab(num_domains)
        for islab in range(self.nslabs):
            coords = self.coord_map.getIndices(islab)
            lb = ones(self.grid.ndims, dtype=int)
            ub = ones(self.grid.ndims, dtype=int)
            for idim in range(self.grid.ndims):
                lb[idim] = coords[idim] * domain_size[idim]
                ub[idim] = (coords[idim] + 1) * domain_size[idim]
                if coords[idim] == num_domains[idim] - 1:
                    ub[idim] = self.grid.numCells[idim]
            self.slabs.append(Slab(lb, ub))

        logger.debug("Domain decomposed into slabs:")
        for slab in self.slabs:
            logger.debug(
                "lb: " + str(slab.lowerBounds) + ", ub: " + str(slab.upperBounds)
            )

    def __perform_geometry_biased_decomp(self):
        # reset slabs, get factors
        self.slabs = []
        factors = self.__prime_factors(self.nslabs)
        logger.debug(
            str(self.nslabs) + " slabs broken into prime factors: " + str(factors)
        )

        # construct int array of entire domain for whether cells have geometry or not
        has_geometry_domain = zeros(self.grid.numCells, dtype=int)
        for i in range(self.grid.numCells[0]):
            for j in range(self.grid.numCells[1]):
                for k in range(self.grid.numCells[2]):
                    if self.grid.cells[(i,j,k)].has_geometry:
                        has_geometry_domain[i, j, k] = 1

        # determine how to slice
        domain_size = copy(self.grid.numCells)
        num_domains = ones(self.grid.ndims,dtype=int)
        for f in factors:
            ind = argmax(domain_size)
            domain_size[ind] /= f
            num_domains[ind] *= f

        logger.debug("GeometryBiasedDecomp: domain_size = " + str(domain_size))
        logger.debug("GeometryBiasedDecomp: num_domains = " + str(num_domains))

        # get slices based on geometry
        domain_edges = [zeros(nd+1) for nd in num_domains]
        for idim in range(3):
            # get cumulative sum along idim axis for num cells with geom
            idim_dist_cum = array(cumsum(
                sum(has_geometry_domain, axis=((idim + 1) % 3, (idim + 2) % 3))
            ),dtype=float)
            idim_dist_cum /= max(idim_dist_cum)
            frac = 1./num_domains[idim]
            for islice in array(range(num_domains[idim]))+1:
                domain_edges[idim][islice] = argmax(idim_dist_cum >= islice*frac)
            domain_edges[idim][-1] += 1

        # generate slabs
        self.coord_map = IndexSlab(num_domains)
        for islab in range(self.nslabs):
            coords = self.coord_map.getIndices(islab)
            lb = ones(self.grid.ndims, dtype=int)
            ub = ones(self.grid.ndims, dtype=int)
            for idim in range(self.grid.ndims):
                lb[idim] = domain_edges[idim][coords[idim]]
                ub[idim] = domain_edges[idim][coords[idim]+1]
            self.slabs.append(Slab(lb, ub))

        logger.debug("Domain decomposed into slabs:")
        for slab in self.slabs:
            logger.debug(
                "lb: " + str(slab.lowerBounds) + ", ub: " + str(slab.upperBounds)
            )
    
    def __merge_with_nearest_smallest_neighbor(self,slab:Slab) -> None:
        """Merge slab with its nearest, smallest neighbor

        Args:
            slab (Slab): slab to merge
        """
        # need to find nearest neighbors, then find the smallest one and merge, then
        #  refine grid again
        
        return


    def plot(self, axes=None, plot=False, by_index=False):
        if self.grid.ndims == 3:
            if not axes:
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_aspect("equal")
            else:
                ax = axes

            for slab in self.slabs:
                if by_index:
                    PlotRectangles.draw_3D_box(ax, slab)
                else:
                    PlotRectangles.draw_3D_box(
                        ax,
                        Slab(
                            self.grid.get_pos_at_ind(slab.lowerBounds),
                            self.grid.get_pos_at_ind(slab.upperBounds),
                        ),
                    )

        elif self.grid.ndims == 2:
            if not axes:
                fig = plt.figure()
                ax = fig.subplot(111)
                ax.set_aspect("equal")
            else:
                ax = axes

            for slab in self.slabs:
                PlotRectangles.draw_2D_box(ax, slab)

        else:
            if not axes:
                fig = plt.figure()
                ax = fig.subplot(111)
                ax.set_aspect("equal")
            else:
                ax = axes

            for slab in self.slabs:
                PlotRectangles.draw_1D_box(ax, slab)

        if plot:
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
    def __init__(self, file):
        self.stl_mesh = mesh.Mesh.from_file(file)

    def get_vertices(self):
        return self.stl_mesh.points.reshape([-1, 3])

    def get_triangles(self):
        return self.stl_mesh.points

    def plot(self, plot=False):
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

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

    def check_tricube_intersection(self, v0, v1, v2, h):
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


class PlotRectangles:
    def draw_3D_box(ax, slab: Slab, draw_surfaces=False):
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

    def draw_2D_box(ax, slab):
        x_range = slab.get_range(0)
        y_range = slab.get_range(1)

        xx, yy = meshgrid(x_range, y_range)
        ax.plot(xx[0], yy[0], color="r")
        ax.plot(xx[0], yy[1], color="r")
        ax.plot(xx[1], yy[0], color="r")
        ax.plot(xx[1], yy[1], color="r")

    def draw_1D_box(ax, slab):
        x_range = slab.get_range(0)

        xx, yy = meshgrid(x_range, array([-1, 1]))
        ax.plot(xx[0], yy[0], color="r")
        ax.plot(xx[0], yy[1], color="r")
        ax.plot(xx[1], yy[0], color="r")
        ax.plot(xx[1], yy[1], color="r")
