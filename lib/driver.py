import stl
from grid import Grid, Decomp, Geometry


def runTest():
    dim = 3
    nx = [30, 30, 30]
    sp = [0.0, 0.0, 0.0]
    ep = [1.0, 1.0, 1.0]

    show_plots = True

    # create geometry
    geometry_names = ["coupler_reduced", "Moon", "Torus_reduced", "C100_reduced"]
    geom_name = geometry_names[0]
    geom = Geometry(geom_name + ".stl")

    # create grid
    # grid = Grid((10,15,20),(-1.0,-1.0,-1.0),(1.0,2.0,1.0))
    if geom_name == "Moon":
        grid = Grid((30, 8, 60), geometry=geom)  # with moon
    elif geom_name == "Torus_reduced":
        grid = Grid((35, 35, 15), geometry=geom)
    elif geom_name == "C100_reduced":
        grid = Grid((15, 10, 50), geometry=geom)
    elif geom_name == "coupler_reduced":
        grid = Grid((40, None, None), geometry=geom)

    decomp = Decomp(grid, 64, geometry_biased=False)
    if (show_plots):
        ax = geom.plot(plot=False)
        decomp.plot(axes=ax, plot=True)

    decomp = Decomp(grid, 64, geometry_biased=True)
    if (show_plots):
        ax = geom.plot(plot=False)
        decomp.plot(axes=ax, plot=True)

    # refine and plot
    decomp.refine_empty(refill_empty=True)
    if (show_plots):
        ax = geom.plot(plot=False)
        decomp.plot(axes=ax, plot=True)

    # try to merge any very small slabs
    decomp.refine_small()
    if (show_plots):
        ax = geom.plot(plot=False)
        decomp.plot(axes=ax, plot=True)

    decomp.diagnostics(plot=True)

if __name__ == "__main__":
    runTest()
