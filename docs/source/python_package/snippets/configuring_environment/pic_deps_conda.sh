#!/bin/bash
# BEGIN-PIC-DEPS-CONDA
# Conda variant: conda-forge supplies everything except PNGwriter.
# Install the rest into the prefix and build PNGwriter against it.
export PIC_LIBS="$HOME/pic-libs/conda"
micromamba create -y -p "$PIC_LIBS" -c conda-forge \
    'gxx_linux-64>=10' 'cmake>=3.28' openmpi 'boost>=1.87' \
    hdf5='*=mpi*' adios2='*=mpi*' openpmd-api fftw c-blosc2 libpng zlib
micromamba run -p "$PIC_LIBS" bash -c '
    cmake -S pngwriter -B pngwriter-build \
        -DCMAKE_INSTALL_PREFIX="$PIC_LIBS/pngwriter/0.7.0" \
        -DCMAKE_PREFIX_PATH="$PIC_LIBS" \
        -DPNGwriter_USE_FREETYPE=OFF &&
    cmake --build pngwriter-build -j "$(nproc)" --target install'
# END-PIC-DEPS-CONDA
