# slagg

Begin by creating a conda environment and installing pythonocc:

conda create --name pyoccenv
conda activate pyoccenv
conda install -c conda-forge pythonocc-core

NOTE: From now on, always use conda install -c conda-forge if more packages need to be installed to avoid conflicts

Then to run a solve:

cd path/to/adjhomopt/rectcov/lib
python main.py mycad.stp x y z delta n

Where...
mycad.stp is a geometry file (the repo has a sample file called C100_export.stp)
x is the x-dimension of the grid to be superimposed on the geometry
y is the y-dimension of the grid to be superimposed on the geometry
z is the z-dimension of the grid to be superimposed on the geometry
delta is the uniform length, width and height of the grid cells
n is the target number of rectangles to be retuned in the final decomposition
