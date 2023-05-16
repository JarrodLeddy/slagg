import stl
from slagg.grid import Grid, Decomp, Geometry


def test_moon():
    dim = 3

    # create geometry
    geom_name = "Moon"
    geom = Geometry("./stl_files/Moon.stl")

    # create grid
    grid = Grid((20, 6, 40), geometry=geom)  # with moon

    decomp = Decomp(grid, 32, geometry_biased=False)
    decomp = Decomp(grid, 32, geometry_biased=True)
    decomp.refine_empty(refill_empty=True)
    decomp.refine_small()
    decomp.diagnostics(plot=False)

