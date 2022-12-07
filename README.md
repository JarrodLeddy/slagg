# slagg

Begin by creating a conda environment and installing pythonocc:

```
conda create --name pyoccenv
conda activate pyoccenv
conda install -c conda-forge pythonocc-core
conda install -c conda-forge numpy-stl
conda install -c conda-forge matplotlib
```
NOTE: From now on, always use conda install -c conda-forge if more packages need to be installed to avoid conflicts

Then to run a solve:

```
cd path/to/slagg/lib
python main.py -f mycad.stp -x nx -y ny -z nz -d delta -n nrects
```

Where...

* mycad.stp is a STEP geometry file (the repo has a sample file called C100_export.stp)
* nx is the x-dimension of the grid to be superimposed on the geometry
* ny is the y-dimension of the grid to be superimposed on the geometry
* nz is the z-dimension of the grid to be superimposed on the geometry
* delta is the uniform length, width and height of the grid cells
* nrects is the target number of rectangles to be retuned in the final decomposition

The flags and their corresponding arguments may be added in whatever order desired
