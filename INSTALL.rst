.. _install-dependencies:

.. seealso::

   You will need to understand how to use `the terminal <http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html>`_, what are `environment variables <https://unix.stackexchange.com/questions/44990/what-is-the-difference-between-path-and-ld-library-path/45106#45106>`_ and please read our :ref:`compiling introduction <install-source>`.

.. note::

   If you are a scientific user at a super computing facility we might have already prepared a software setup for you.
   See the :ref:`following chapter <install-profile>` if you can skip this step fully or in part by loading existing modules on those systems.

Dependencies
============

.. sectionauthor:: Axel Huebl, Klaus Steiniger, Sergei Bastrakov, Rene Widera

Overview
--------

.. figure:: libraryDependencies.png
   :alt: overview of PIConGPU library dependencies

   Overview of inter-library dependencies for parallel execution of PIConGPU on a typical HPC system. Due to common binary incompatibilities between compilers, MPI and boost versions, we recommend to organize software with a version-aware package manager such as `spack <https://github.com/spack/spack>`_ and to deploy a hierarchical module system such as `lmod <https://github.com/TACC/Lmod>`_.
   An Lmod example setup can be found `here <https://github.com/ComputationalRadiationPhysics/compileNode>`_.

Requirements
------------

Mandatory
^^^^^^^^^

Compiler
""""""""
- C++17 supporting compiler, e.g. GCC 9+ or Clang 11+
- if you want to build for Nvidia GPUs, check the `CUDA supported compilers <https://gist.github.com/ax3l/9489132>`_ page
- *note:* be sure to build all libraries/dependencies with the *same* compiler version
- *Debian/Ubuntu:*

  - ``sudo apt-get install gcc-9 g++-9 build-essential``
  - ``sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9``
- *Arch Linux:*

  - ``sudo pacman --sync base-devel``
  - if the installed version of **gcc** is too new, `compile an older gcc <https://gist.github.com/slizzered/a9dc4e13cb1c7fffec53>`_
- *Spack:*

  - ``spack install gcc@12.2.0``
  - make it the default in your `packages.yaml <http://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration>`_ or *suffix* `all following <http://spack.readthedocs.io/en/latest/features.html#simple-package-installation>`_ ``spack install`` commands with a *space* and ``%gcc@12.2.0``

CMake
"""""
- 3.22.0 or higher
- *Debian/Ubuntu:* ``sudo apt-get install cmake file cmake-curses-gui``
- *Arch Linux:* ``sudo pacman --sync cmake``
- *Spack:* ``spack install cmake``

MPI 2.3+
""""""""
- **OpenMPI** 1.7+ / **MVAPICH2** 1.8+ or similar
- for running on Nvidia GPUs, perform a `GPU aware MPI install <https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/>`_ *after* installing CUDA
- *Debian/Ubuntu:* ``sudo apt-get install libopenmpi-dev``
- *Arch Linux:* ``sudo pacman --sync openmpi``
- *Spack:*

  - *GPU support:* ``spack install openmpi+cuda``
  - *CPU only:* ``spack install openmpi``
- *environment:*

  - ``export MPI_ROOT=<MPI_INSTALL>``
  - as long as CUDA awareness (``openmpi+cuda``) is missing: ``export OMPI_MCA_mpi_leave_pinned=0``

Boost
"""""
- 1.74.0+ (``program_options``, ``atomic`` and header-only libs)
- *Debian/Ubuntu:* ``sudo apt-get install libboost-program-options-dev libboost-atomic-dev``
- *Arch Linux:* ``sudo pacman --sync boost``
- *Spack:* ``spack install boost +program_options +atomic``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``cd ~/src``
  - ``curl -Lo boost_1_74_0.tar.gz https://boostorg.jfrog.io/artifactory/main/release/1.74.0/source/boost_1_74_0.tar.gz``
  - ``tar -xzf boost_1_74_0.tar.gz``
  - ``cd boost_1_74_0``
  - ``./bootstrap.sh --with-libraries=atomic,program_options --prefix=$HOME/lib/boost``
  - ``./b2 cxxflags="-std=c++17" -j4 && ./b2 install``
- *environment:* (assumes install from source in ``$HOME/lib/boost``)

  - ``export CMAKE_PREFIX_PATH=$HOME/lib/boost:$CMAKE_PREFIX_PATH``

git
"""
- not required for the code, but for our workflows
- 1.7.9.5 or `higher <https://help.github.com/articles/https-cloning-errors>`_
- *Debian/Ubuntu:* ``sudo apt-get install git``
- *Arch Linux:* ``sudo pacman --sync git``
- *Spack:* ``spack install git``

rsync
"""""
- not required for the code, but for our workflows
- *Debian/Ubuntu:* ``sudo apt-get install rsync``
- *Arch Linux:* ``sudo pacman --sync rsync``
- *Spack:* ``spack install rsync``

alpaka 1.1.X
""""""""""""""""
- `alpaka <https://github.com/alpaka-group/alpaka>`_ is included in the PIConGPU source code

mallocMC 2.6.0crp-dev
"""""""""""""""""""""
- only required for CUDA and HIP backends
- `mallocMC <https://github.com/ComputationalRadiationPhysics/mallocMC>`_ is included in the PIConGPU source code

.. _install-dependencies-picongpu:

PIConGPU Source Code
^^^^^^^^^^^^^^^^^^^^

- ``git clone https://github.com/ComputationalRadiationPhysics/picongpu.git $HOME/src/picongpu``

  - *optional:* update the source code with ``cd $HOME/src/picongpu && git fetch && git pull``
  - *optional:* change to a different branch with ``git branch`` (show) and ``git checkout <BranchName>`` (switch)
- *environment*:

  - ``export PICSRC=$HOME/src/picongpu``
  - ``export PIC_EXAMPLES=$PICSRC/share/picongpu/examples``
  - ``export PATH=$PATH:$PICSRC``
  - ``export PATH=$PATH:$PICSRC/bin``
  - ``export PATH=$PATH:$PICSRC/src/tools/bin``
  - ``export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH``

Optional Libraries
^^^^^^^^^^^^^^^^^^

CUDA
""""
- `11.0.0+ <https://developer.nvidia.com/cuda-downloads>`_
- required if you want to run on Nvidia GPUs
- *Debian/Ubuntu:* ``sudo apt-get install nvidia-cuda-toolkit``
- *Arch Linux:* ``sudo pacman --sync cuda``
- *Spack:* ``spack install cuda``
- at least one **CUDA** capable **GPU**
- *compute capability*: ``sm_60`` or higher
- `full list <https://developer.nvidia.com/cuda-gpus>`_ of CUDA GPUs and their *compute capability*
- `More <http://www.olcf.ornl.gov/summit/>`_ is always `better <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html>`_. Especially, if we are talking GPUs :-)
- *environment:*

  - ``export CUDA_ROOT=<CUDA_INSTALL>``

If you do not install the following libraries, you will not have the full amount of PIConGPU plugins.
We recommend to install at least **pngwriter** and **openPMD**.

libpng
""""""
- 1.2.9+ (requires *zlib*)
- *Debian/Ubuntu dependencies:* ``sudo apt-get install libpng-dev``
- *Arch Linux dependencies:* ``sudo pacman --sync libpng``
- *Spack:* ``spack install libpng``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``cd ~/src``
  - ``curl -Lo libpng-1.6.34.tar.gz ftp://ftp-osl.osuosl.org/pub/libpng/src/libpng16/libpng-1.6.34.tar.gz``
  - ``tar -xf libpng-1.6.34.tar.gz``
  - ``cd libpng-1.6.34``
  - ``CPPFLAGS=-I$HOME/lib/zlib/include LDFLAGS=-L$HOME/lib/zlib/lib ./configure --enable-static --enable-shared --prefix=$HOME/lib/libpng``
  - ``make``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/libpng``)

  - ``export PNG_ROOT=$HOME/lib/libpng``
  - ``export CMAKE_PREFIX_PATH=$PNG_ROOT:$CMAKE_PREFIX_PATH``

pngwriter
"""""""""
- 0.7.0+ (requires *libpng*, *zlib*, and optional *freetype*)
- *Spack:* ``spack install pngwriter``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``git clone -b 0.7.0 https://github.com/pngwriter/pngwriter.git ~/src/pngwriter/``
  - ``cd ~/src/pngwriter``
  - ``mkdir build && cd build``
  - ``cmake -DCMAKE_INSTALL_PREFIX=$HOME/lib/pngwriter ..``
  - ``make install``

- *environment:* (assumes install from source in ``$HOME/lib/pngwriter``)

  - ``export CMAKE_PREFIX_PATH=$HOME/lib/pngwriter:$CMAKE_PREFIX_PATH``

openPMD API
"""""""""""
- optional, but strongly recommended as most PIConGPU output requires it
- 0.15.0+
- *Spack*: ``spack install openpmd-api``
- For usage in PIConGPU, the openPMD API must have been built either with support for ADIOS2 or HDF5 (or both).
  When building the openPMD API from source (described below), these dependencies must be built and installed first.

  - For ADIOS2, CMake build instructions can be found in the `official documentation <https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html>`_.
    Besides compression, the default configuration should generally be sufficient, the ``CMAKE_INSTALL_PREFIX`` should be set to a fitting location. Compression with ``c-blosc`` is described below.
  - For HDF5, CMake build  instructions can be found in the `official documentation <https://support.hdfgroup.org/HDF5/release/cmakebuild.html>`_.
    The parameters ``-DHDF5_BUILD_CPP_LIB=OFF -DHDF5_ENABLE_PARALLEL=ON`` are required, the ``CMAKE_INSTALL_PREFIX`` should be set to a fitting location.
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``git clone -b 0.15.0 https://github.com/openPMD/openPMD-api.git ~/src/openPMD-api``
  - ``cd ~/src/openPMD-api``
  - ``mkdir build && cd build``
  - ``cmake .. -DopenPMD_USE_MPI=ON -DCMAKE_INSTALL_PREFIX=~/lib/openPMD-api``
    Optionally, specify the parameters ``-DopenPMD_USE_ADIOS2=ON -DopenPMD_USE_HDF5=ON``. Otherwise, these parameters are set to ``ON`` automatically if CMake detects the dependencies on your system.
  - ``make -j $(nproc) install``
- environment:* (assumes install from source in ``$HOME/lib/openPMD-api``)

  - ``export CMAKE_PREFIX_PATH="$HOME/lib/openPMD-api:$CMAKE_PREFIX_PATH"``
- If PIConGPU is built with openPMD output enabled, the JSON library
  nlohmann_json will automatically be used, found in the ``thirdParty/``
  directory.
  By setting the CMake parameter ``PIC_nlohmann_json_PROVIDER=extern``, CMake
  can be instructed to search for an installation of nlohmann_json externally.
  Refer to LICENSE.md for further information.

c-blosc for openPMD API with ADIOS2
"""""""""""""""""""""""""""""""""""
- not a direct dependency of PIConGPU, but an optional dependency for openPMD API with ADIOS2; installation is described here since it is lacking in documentation elsewhere
- general purpose compressor, used in ADIOS2 for in situ data reduction
- *Debian/Ubuntu:* ``sudo apt-get install libblosc-dev``
- *Arch Linux:* ``sudo pacman --sync blosc``
- *Spack:* ``spack install c-blosc``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``git clone -b v1.21.1 https://github.com/Blosc/c-blosc.git ~/src/c-blosc/``
  - ``cd ~/src/c-blosc``
  - ``mkdir build && cd build``
  - ``cmake -DCMAKE_INSTALL_PREFIX=$HOME/lib/c-blosc -DPREFER_EXTERNAL_ZLIB=ON ..``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/c-blosc``)

  - ``export BLOSC_ROOT=$HOME/lib/c-blosc``
  - ``export CMAKE_PREFIX_PATH=$BLOSC_ROOT:$CMAKE_PREFIX_PATH``

ISAAC
"""""
- 1.6.0+
- requires *boost* (header only), *IceT*, *Jansson*, *libjpeg* (preferably *libjpeg-turbo*), *libwebsockets* (only for the ISAAC server, but not the plugin itself)
- enables live in situ visualization, see more here `Plugin description <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/Plugin%3A-ISAAC>`_
- *Spack:* ``spack install isaac``
- *from source:* build the *in situ library* and its dependencies as described in `ISAAC's INSTALL.md <https://github.com/ComputationalRadiationPhysics/isaac/blob/master/INSTALL.md>`_
- *environment:* set environment variable ``CMAKE_PREFIX_PATH`` for each dependency and the ISAAC in situ library

FFTW3
"""""
- required for Shadowgraphy plugin
- *from tarball:*

  - ``mkdir -p ~/src ~/lib``
  - ``cd ~/src``
  - ``wget -O fftw-3.3.10.tar.gz http://fftw.org/fftw-3.3.10.tar.gz``
  - ``tar -xf fftw-3.3.10.tar.gz``
  - ``cd fftw-3.3.10``
  - ``./configure --prefix="$FFTW_ROOT"``
  - ``make``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/fftw-3.3.10``)

  - ``export FFTW3_ROOT =$HOME/lib/fftw-3.3.10
  - ``export LD_LIBRARY_PATH=$FFTW3_ROOT/lib:$LD_LIBRARY_PATH``
