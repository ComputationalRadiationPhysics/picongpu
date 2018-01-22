png2gas
================================================================

### About

png2gas is a small tool that allows you to create a libSplash/HDF5 input file
when using the gasFromHdf5 gas initialization option in PIConGPU.
The density information is created from a png file.


### Install

Required libraries:
 - **cmake** 2.8.12.2 or higher
 - **OpenMPI** 1.4 or higher
 - **boost** 1.47.0 or higher ("program options")
 - **PNGwriter** 0.7.0 or higher ([GitHub project](https://github.com/pngwriter/pngwriter))
 - **libSplash** (requires *hdf5*)
 - **hdf5** >= 1.8.6, standard shared version (no c++, enable parallel)


### Usage

png2gas requires a png file with the same size (width, height) as the density data (y, x)
you want to create. Run `png2gas --help` for detailed usage information.

Valid density input images are greyscale PNGs. The **Value** component of the image in
HSV colorspace is used for the normalized density as a 32bit float value in [0.0,1.0].
Black (Value = 0.0) in the input image is considered no density/vacuum, white is used accordingly.

