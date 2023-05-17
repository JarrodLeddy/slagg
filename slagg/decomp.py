from slagg.utils import PlotRectangles, Slab, IndexSlab
from slagg.grid import Grid, Cell
from slagg.geometry import Geometry

from numpy import (
    array,
    ndarray,
    max,
    min,
    sum,
    argmax,
    mean,
    std,
    append,
    ones,
    cumsum,
    copy,
    zeros,
    flip,
    argsort,
)
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


class Decomp:
    slabs = []
    nslabs = 1  # desired number of slabs, not always equal to len(slabs)

    def __init__(self, grid: Grid, nslabs: int, geometry_biased=True):
        """Initialize the Decomposition

        Args:
            grid (Grid): The grid object on which the decomp should be performed.
            nslabs (int): Desired number of slabs for the decomposition.
            geometry_biased (bool, optional): Improve original decomp with geometry-biased algorithm. Defaults to True.
        """
        self.nslabs = nslabs
        self.grid = grid

        # do regular decomposition
        self.__perform_regular_decomp()

        # save for diagnostics later
        self.initial_volumes = array([i.get_volume() for i in self.slabs])
        self.initial_volume = sum(self.initial_volumes)
        self.__initial_geometry_diagnostics()

        # now do geometry-biased decomp (should be a better starting point)
        if geometry_biased:
            self.__perform_geometry_biased_decomp()

    def diagnostics(self, plot=False):
        """Print (and optionally plot) diagnostics data about the level
        of improvement between the current decomp and the original standard decomp

        Args:
            plot (bool, optional): Whether to show the diagnostic plots. Defaults to False.
        """
        # first lets look at the overall gain in memory allocation
        volumes = array([i.get_volume() for i in self.slabs])
        total_volume = sum(volumes)
        logger.info("\n ========== Summary ==========\n")
        logger.info("-- Memory --")
        logger.info(f"Initial domain volume: {self.initial_volume}")
        logger.info(f"Final domain volume: {total_volume}")
        logger.info(
            f"Results in {total_volume/self.initial_volume*100:.1f}% memory usage.\n"
        )

        # get info about cells with geometry in the slab
        slab_geom_volume = zeros(len(self.slabs))
        for islab, slab in enumerate(self.slabs):
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
            slab_geom_volume[islab] = sum(has_geometry_slab)

        logger.info(f"\n-- Efficiency --")
        logger.info(f"Before Refinement:")
        logger.info(f"{self.nslabs} total domains")
        logger.info(f"Average {mean(self.initial_volumes):.1f} cells per domain")
        logger.info(
            f"Average {mean(self.initial_slab_geom_volume):.1f} cells per domain containing geometry"
        )
        logger.info(
            f"Largest slab has {max(self.initial_slab_geom_volume)} cells containing geometry"
        )
        logger.info(
            f"{std(self.initial_slab_geom_volume):.1f} standard deviation of cells per domain containing geometry"
        )

        logger.info(f"\nAfter Refinement:")
        logger.info(f"{len(self.slabs)} total domains")
        logger.info(f"Average {mean(volumes):.1f} cells per domain")
        logger.info(
            f"Average {mean(slab_geom_volume):.1f} cells per domain containing geometry"
        )
        logger.info(
            f"Largest slab has {max(slab_geom_volume)} cells containing geometry"
        )
        logger.info(
            f"{std(slab_geom_volume):.1f} standard deviation of cells per domain containing geometry"
        )

        if plot:
            # get histogram data
            vrange = (
                min(array([self.initial_volumes, volumes])),
                max(array([self.initial_volumes, volumes])),
            )
            gvrange = (
                min(array([self.initial_slab_geom_volume, slab_geom_volume])),
                max(array([self.initial_slab_geom_volume, slab_geom_volume])),
            )
            pad = 5
            hist_resolution = 3

            fig = plt.figure(figsize=(10, 8))
            # before refinement
            ax = fig.add_subplot(221)
            ax.hist(
                self.initial_volumes,
                bins=self.nslabs // hist_resolution,
                range=vrange,
                edgecolor="black",
            )
            ax.set_xlabel("cells per domain")
            ax.set_ylabel("number of domains")

            ax.annotate(
                "Total Cells",
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

            ax.annotate(
                "Basic\nDecomp",
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

            ax = fig.add_subplot(222)
            ax.hist(
                self.initial_slab_geom_volume,
                bins=self.nslabs // hist_resolution,
                range=gvrange,
                edgecolor="black",
            )
            ax.set_xlabel("cells with geometry per domain")
            ax.set_ylabel("number of domains")

            ax.annotate(
                "Cells With Geometry",
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

            # after refinement
            ax = fig.add_subplot(223)
            ax.hist(
                volumes,
                bins=self.nslabs // hist_resolution,
                range=vrange,
                edgecolor="black",
            )
            ax.set_xlabel("cells per domain")
            ax.set_ylabel("number of domains")

            ax.annotate(
                "Refined\nDecomp",
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

            ax = fig.add_subplot(224)
            ax.hist(
                slab_geom_volume,
                bins=self.nslabs // hist_resolution,
                range=gvrange,
                edgecolor="black",
            )
            ax.set_xlabel("cells with geometry per domain")
            ax.set_ylabel("number of domains")

            # display
            fig.tight_layout()
            fig.subplots_adjust(left=0.15, top=0.95)
            plt.show()

    def refine(self, plot=False) -> None:
        """One-stop call to apply all refinement techniques plus print diagnostics

        Args:
            plot (bool, optional): Whether to show diagnostics plots. Defaults to False.
        """
        self.refine_empty()
        self.refine_small()
        self.diagnostics(plot=False)

    def refine_small(self) -> None:
        """Checks for outlyingly small slabs and attempts to merge them with neighbors"""
        # check if volume is more than 2 standard deviations from the average volume
        #  and if so try to merge with neighbor
        vols = array([islb.get_volume() for islb in self.slabs])
        stdvol = std(vols)
        avgvol = mean(vols)
        for slb in self.slabs:
            if slb.get_volume() < avgvol - 2 * stdvol:
                logger.info(
                    str(slb)
                    + " is %.2f" % ((avgvol - slb.get_volume()) / stdvol)
                    + " standard deviations below the average volume. Attempting to merge with neighbor."
                )
                merged = self.__merge_with_nearest_smallest_neighbor(slb)
                if merged:
                    self.refine_empty()
                    logger.info("Merged successfully")
                else:
                    logger.info("Merge failed")
        return

    def refine_empty(self, refill_empty=True) -> None:
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
            # shoudl always be possible to split slabs, so this should never be infinite
            while self.nslabs != len(self.slabs):
                self.__refill_empty_slabs()
                self.__squeeze_empty()

        return

    def __refill_empty_slabs(self) -> None:
        """If the current number of slabs is less than the number the user asked for,
        this method will split the largest slabs until the desired number of slabs is
        obtained.
        """
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
            "Largest slabs split to create " + str(len(self.slabs)) + " total slabs."
        )
        return

    def __squeeze_empty(self) -> None:
        """Removes all cells that do not have geometry from decomp slabs, where possible, by
        shrinking the bounds on the Slabs. Some slabs may end up empty (ie. containing
        no cells at all) and they are removed from the decomp slabs list
        """
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
            "After initial refinement, " + str(len(self.slabs)) + " slabs remaining.\n"
        )
        return

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
        """Performs a standard decomp. The resulting number of slabs will be
        exactly the number requested by the user in initialization. The domain
        slabs will be as cubic and equally-sized as possible. This uses prime
        factorization to split the largest dimensions the most.
        """
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
        """Performs a decomposition that is biased such that the domain slabs will
        contain as equal number of cells that contain geometry. The resulting number
        of slabs will be exactly the number requested by the user in initialization.
        This uses prime factorization to split the largest dimensions the most, and
        uses the marginlized cumulative distributions of the geometry cell count to
        equally split.
        """
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
                    if self.grid.cells[(i, j, k)].has_geometry:
                        has_geometry_domain[i, j, k] = 1

        # determine how to slice
        domain_size = copy(self.grid.numCells)
        num_domains = ones(self.grid.ndims, dtype=int)
        for f in factors:
            ind = argmax(domain_size)
            domain_size[ind] /= f
            num_domains[ind] *= f

        logger.debug("GeometryBiasedDecomp: domain_size = " + str(domain_size))
        logger.debug("GeometryBiasedDecomp: num_domains = " + str(num_domains))

        # get slices based on geometry
        domain_edges = [zeros(nd + 1) for nd in num_domains]
        for idim in range(3):
            # get cumulative sum along idim axis for num cells with geom
            idim_dist_cum = array(
                cumsum(sum(has_geometry_domain, axis=((idim + 1) % 3, (idim + 2) % 3))),
                dtype=float,
            )
            idim_dist_cum /= max(idim_dist_cum)
            frac = 1.0 / num_domains[idim]
            for islice in array(range(num_domains[idim])) + 1:
                domain_edges[idim][islice] = argmax(idim_dist_cum >= islice * frac)
            domain_edges[idim][-1] += 1

        # generate slabs
        self.coord_map = IndexSlab(num_domains)
        for islab in range(self.nslabs):
            coords = self.coord_map.getIndices(islab)
            lb = ones(self.grid.ndims, dtype=int)
            ub = ones(self.grid.ndims, dtype=int)
            for idim in range(self.grid.ndims):
                lb[idim] = domain_edges[idim][coords[idim]]
                ub[idim] = domain_edges[idim][coords[idim] + 1]
            self.slabs.append(Slab(lb, ub))

        logger.debug("Domain decomposed into slabs:")
        for slab in self.slabs:
            logger.debug(
                "lb: " + str(slab.lowerBounds) + ", ub: " + str(slab.upperBounds)
            )

    def __merge_with_nearest_smallest_neighbor(self, slab: Slab) -> bool:
        """Merge slab with its nearest, smallest neighbor

        Args:
            slab (Slab): slab to merge

        Returns:
            (bool): whether a merge occurred or not
        """
        # need to find nearest neighbors, then find the smallest one and merge, then
        #  refine grid again
        neighbors = []
        for i, slb in enumerate(self.slabs):
            if slab.neighbors(slb) and slab != slb:
                neighbors.append([i, slb])

        # now we have list of neighbors, lets try merging smallest
        vols = array([i[1].get_volume() for i in neighbors])
        sinds = argsort(vols)
        for ind in sinds:
            i, islab = neighbors[ind]
            merged_slab = slab.union(islab)
            # now check to see if this merged_slab overlaps with any other slabs, if so reject
            accept_new_slab = False
            num_overlap = 0
            for j, jslab in enumerate(self.slabs):
                if not merged_slab.intersection(jslab).is_empty():
                    num_overlap += 1
                    # slab will intersect with both initial slabs, but if more then reject
                    if num_overlap > 2:
                        break
            if num_overlap > 2:
                # go to next neighbor
                continue
            else:
                # accept the merged slab (by replacing the provided slab), delete the other two
                self.slabs[i] = merged_slab
                slab.set_empty()
                return True
        return False

    def __initial_geometry_diagnostics(self) -> None:
        """Prints (and optionally plots) diagnostic information about the quality
        of the decomposition.
        """
        self.initial_slab_geom_volume = zeros(len(self.slabs))
        for islab, slab in enumerate(self.slabs):
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
            self.initial_slab_geom_volume[islab] = sum(has_geometry_slab)
        return

    def plot(self, axes=None, plot=False, by_index=False):
        """Plots the decomposition as rectangles

        Args:
            axes (axis, optional): Axis on which to plot. Defaults to None.
            plot (bool, optional): Whether or not to show the plot at the end. Defaults to False.
            by_index (bool, optional): Whether the axees should be in index or physical coords. Defaults to False.

        Returns:
            _type_: _description_
        """
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

    def __prime_factors(self, n: int) -> list:
        """Finds all of the prime factors of an integer

        Args:
            n (int): number of factorize

        Returns:
            list: contains all the prime factors of n
        """
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
