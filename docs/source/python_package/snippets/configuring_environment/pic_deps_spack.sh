#!/bin/bash
# BEGIN-PIC-DEPS-SPACK
# Spack variant: build the same set and point the profile at the Spack prefix.
. /path/to/spack/share/spack/setup-env.sh
export PIC_LIBS="$HOME/pic-libs/spack"
spack install --prefix="$PIC_LIBS/boost/1.87.0" boost@1.87.0 +program_options
spack install --prefix="$PIC_LIBS/hdf5/2.0.0" hdf5@2.0.0 +mpi +cxx
spack install --prefix="$PIC_LIBS/adios2/2.11.0" adios2@2.11.0 +mpi +hdf5 +blosc
spack install --prefix="$PIC_LIBS/openpmd-api/0.17.1" openpmd-api@0.17.1 +mpi +hdf5 +adios2
spack install --prefix="$PIC_LIBS/fftw/3.3.10" fftw@3.3.10
spack install --prefix="$PIC_LIBS/c-blosc2/2.22.0" c-blosc2@2.22.0
spack install --prefix="$PIC_LIBS/libpng/1.6.34" libpng@1.6.34
spack install --prefix="$PIC_LIBS/pngwriter/0.7.0" pngwriter@0.7.0 ^libpng
# END-PIC-DEPS-SPACK
