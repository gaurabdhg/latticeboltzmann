# latticeboltzmann
trying out 2D and 3D lattice boltzmann studies over different obstacles with a van karman vortex study.

\n todos:
* improve support for read files. \<implemented but needs debugs\>
* make sure its plotted properly. i.e. when we read a file, the file is shown as base and contours are shown on top of it. Theres already a flag that checks if theres a file or not. just extend it so the plot function can read it.
* maybe introduce an input file that takes the characteristic length, defines 2D or 3D and if 3D D3Q19 or D3Q27 and also has path for the geometry file.
