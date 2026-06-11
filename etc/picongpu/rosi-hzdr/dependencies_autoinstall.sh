#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl, Marco Garten, Klaus Steiniger, Pawel Ordyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
# setup for rosi / HZDR
# last updated: 2026-03-09


# get PIConGPU profile
if [ ! -f "$PIC_PROFILE" ]; then
    printf "Source a profile!\n"
    exit 1
else
    source "$PIC_PROFILE"
fi

set -euf -o pipefail
# create temporary directory for software source files
export SOURCE_DIR="/bigdata/hplsim/development/rosi-picongpu-lib_buildDir"
mkdir -p $SOURCE_DIR

# get openMPI dir for ADIOS2 and openPMD
export OMPI_DIR=$(which mpiexec | sed -e 's%bin/mpiexec%%g')

# Boost
if [ ! -d "$BOOST_ROOT" ]; then
    echo "Installing boost"
    cd $SOURCE_DIR
    curl -L -s -o boost_1_87_0.tar.gz \
        https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
    tar -xzf boost_1_87_0.tar.gz
    cd boost_1_87_0/
    ./bootstrap.sh --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,serialization,system,thread --prefix=$BOOST_ROOT \
        CC=$(which cc) CXX=$(which CC)
    ./b2 cxxflags="-std=c++20" -j 10 && ./b2 install
fi

#   c-blosc
if [ ! -d "$BLOSC_ROOT" ]; then
    echo "Installing c-blosc"
    cd $SOURCE_DIR
    git clone -b v${BLOSC_VERSION} https://github.com/Blosc/c-blosc2.git \
        $SOURCE_DIR/c-blosc
    mkdir c-blosc-build
    cd c-blosc-build
    cmake -DCMAKE_INSTALL_PREFIX=$BLOSC_ROOT \
        -DMPI_C_COMPILER=cc -DMPI_CXX_COMPILER=CC \
        $SOURCE_DIR/c-blosc
    make -j 16 install
fi


#   libpng
if [ ! -d "$LIBPNG_ROOT" ]; then
    echo "Installing libpng"
    cd $SOURCE_DIR
    mkdir -p $SOURCE_DIR/libpng $SOURCE_DIR/libpng
    cd $SOURCE_DIR/libpng
    curl -Lo libpng-1.6.34.tar.gz ftp://ftp-osl.osuosl.org/pub/libpng/src/libpng16/libpng-1.6.34.tar.gz
    tar -xf libpng-1.6.34.tar.gz
    cd libpng-1.6.34
    ./configure --enable-static --enable-shared --prefix=$LIBPNG_ROOT
    make
    make install
fi



# TODO: we need to update PNGwriter to work with a modern cmake

#   PNGwriter
#if [ ! -d "$PNGwriter_ROOT" ]; then
#    echo "Installing PNGwriter"
#    cd $SOURCE_DIR
#    git clone -b 0.7.0 https://github.com/pngwriter/pngwriter.git \
#        $SOURCE_DIR/pngwriter
#    mkdir pngwriter-build
#    cd pngwriter-build
#    cmake --version
#    cmake -DCMAKE_INSTALL_PREFIX=$PNGwriter_ROOT \
#        $SOURCE_DIR/pngwriter
#    make -j 16 install
#fi



#   HDF5
if [ ! -d "$HDF5_ROOT" ]; then
    echo "Installing parallel HDF5..."
    cd $SOURCE_DIR
    wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_${HDF5_VERSION}.tar.gz
    tar -xzf hdf5_${HDF5_VERSION}.tar.gz
    mkdir hdf5_build
    cd hdf5_build
    cmake -DCMAKE_INSTALL_PREFIX=$HDF5_ROOT  -DHDF5_ENABLE_PARALLEL=ON ../hdf5-hdf5_${HDF5_VERSION}/
    make install -j 16

fi



#   ADIOS2
# force usage of MPI and HDF5 and point directly to MPI headers and libraries
if [ ! -d "$ADIOS2_ROOT" ]; then
    echo "Installing adios2"
    cd $SOURCE_DIR
    git clone -b v${ADIOS2_VERSION} https://github.com/ornladios/ADIOS2.git \
        $SOURCE_DIR/adios2
    mkdir $SOURCE_DIR/adios2-build
    cd $SOURCE_DIR/adios2-build

    cmake $SOURCE_DIR/adios2 -DADIOS2_BUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=$ADIOS2_ROOT -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_BZip2=OFF \
        -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON \
        -DMPI_CXX_COMPILER=${MPI_CXX} -DMPI_C_COMPILER=${MPI_CC} \
        -DMPI_CXX_HEADER_DIR=${OMPI_DIR}/include \
        -DMPI_C_HEADER_DIR=${OMPI_DIR}/include \
        -DMPI_mpi_gnu_123_LIBRARY=${OMPI_DIR}/lib/libmpi_gnu_123.so
    make -j 16 && make install
fi

#   openPMD-api
if [ ! -d "OPENPMD_ROOT" ]; then
    echo "Installing openPMD-api"
    cd $SOURCE_DIR
    git clone -b 0.17.0 https://github.com/openPMD/openPMD-api.git \
        $SOURCE_DIR/openpmd-api
    mkdir $SOURCE_DIR/openpmd-api-build
    cd $SOURCE_DIR/openpmd-api-build
    cmake $SOURCE_DIR/openpmd-api \
          -DopenPMD_USE_HDF5=ON  -DopenPMD_USE_ADIOS2=ON \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DMPI_CXX_COMPILER=${MPI_CXX} -DMPI_C_COMPILER=${MPI_CC} \
        -DMPI_CXX_HEADER_DIR=${OMPI_DIR}/include \
        -DMPI_C_HEADER_DIR=${OMPI_DIR}/include \
        -DMPI_mpi_gnu_123_LIBRARY=${OMPI_DIR}/lib/libmpi_gnu_123.so \
        -DCMAKE_INSTALL_PREFIX="$OPENPMD_ROOT"
    make -j 16 install
fi

#    fftw
if [ ! -d "FFTW_ROOT" ]; then
    echo "Installing fftw"
    cd $SOURCE_DIR
    mkdir $SOURCE_DIR/fftw
    cd $SOURCE_DIR/fftw
    wget -O fftw-3.3.10.tar.gz http://fftw.org/fftw-3.3.10.tar.gz
    tar -xf fftw-3.3.10.tar.gz
    cd fftw-3.3.10
    ./configure --prefix=$FFTW_ROOT
    make
    make install
fi
