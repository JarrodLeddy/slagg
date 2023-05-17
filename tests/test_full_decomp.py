import stl
import pathlib
from slagg import Grid, Decomp, Geometry

TESTS_PATH = pathlib.Path(__file__).parent

def test_moon():
    dim = 3

    # create geometry
    file_name = TESTS_PATH / 'stl_files' / 'Moon.stl'
    geom = Geometry(file_name)

    # create grid
    grid = Grid((20, 6, 40), geometry=geom)  # with moon

    decomp = Decomp(grid, 32, geometry_biased=False)
    decomp = Decomp(grid, 32, geometry_biased=True)
    decomp.refine_empty(refill_empty=True)
    decomp.refine_small()
    decomp.diagnostics(plot=False)

