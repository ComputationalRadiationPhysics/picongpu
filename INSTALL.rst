.. _install-dependencies:

.. seealso::

   You will need to understand how to use `the terminal <http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html>`_, what are `environment variables <https://unix.stackexchange.com/questions/44990/what-is-the-difference-between-path-and-ld-library-path/45106#45106>`_ and please read our :ref:`compiling introduction <install-source>`.

.. note::

   If you are a scientific user at a supercomputing facility we might have already prepared a software setup for you.
   See the :ref:`following chapter <install-profile>` if you can skip this step fully or in part by loading existing modules on those systems.

Dependencies
============

.. sectionauthor:: Axel Huebl

Overview
--------

.. figure:: libraryDependencies.png
   :alt: overview of PIConGPU library dependencies

   Overview of inter-library dependencies for parallel execution of PIConGPU on a typical HPC system. Due to common binary incompatibilities between compilers, MPI and boost versions, we recommend to organize software with a version-aware package manager such as `spack <https://github.com/spack/spack>`_ and to deploy a hierarchical module system such as `lmod <https://github.com/TACC/Lmod>`_.
   A Lmod example setup can be found `here <https://github.com/ComputationalRadiationPhysics/compileNode>`_.

Requirements
------------

Mandatory
^^^^^^^^^

gcc
"""
- 4.9 to 5.X (depends on your current `CUDA version <https://gist.github.com/ax3l/9489132>`_)
- *note:* be sure to build all libraries/dependencies with the *same* gcc version
- *Debian/Ubuntu:*
  
  - ``sudo apt-get install gcc-4.9 g++-4.9 build-essential``
  - ``sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9``
- *Arch Linux:*
  
  - ``sudo pacman --sync base-devel``
  - if the installed version of **gcc** is too new, `compile an older gcc <https://gist.github.com/slizzered/a9dc4e13cb1c7fffec53>`_
- *Spack:*
  
  - ``spack install gcc@4.9.4``
  - make it the default in your `packages.yaml <http://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration>`_ or *suffix* `all following <http://spack.readthedocs.io/en/latest/features.html#simple-package-installation>`_ ``spack install`` commands with a *space* and ``%gcc@4.9.4``

CMake
"""""
- 3.10.0 or higher
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

zlib
""""
- *Debian/Ubuntu:* ``sudo apt-get install zlib1g-dev``
- *Arch Linux:* ``sudo pacman --sync zlib``
- *Spack:* ``spack install zlib``
- *from source:*

  - ``./configure --prefix=$HOME/lib/zlib``
  - ``make && make install``
- *environent:* (assumes install from source in ``$HOME/lib/zlib``)

  - ``export ZLIB_ROOT=$HOME/lib/zlib``
  - ``export LD_LIBRARY_PATH=$ZLIB_ROOT/lib:$LD_LIBRARY_PATH``
  - ``export CMAKE_PREFIX_PATH=$ZLIB_ROOT:$CMAKE_PREFIX_PATH``

boost
"""""
- 1.62.0-1.64.0 (``program_options``, ``regex`` , ``filesystem``, ``system``, ``math``, ``serialization`` and header-only libs, optional: ``fiber`` with ``context``, ``thread``, ``chrono``, ``atomic``, ``date_time``)
- download from `http://www.boost.org <http://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.gz/download>`_
- *Debian/Ubuntu:* ``sudo apt-get install libboost-program-options-dev libboost-regex-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-chrono-dev libboost-atomic-dev libboost-date-time-dev libboost-math-dev libboost-serialization-dev libboost-fiber-dev libboost-context-dev``
- *Arch Linux:* ``sudo pacman --sync boost``
- *Spack:* ``spack install boost``
- *from source:*

  - ``./bootstrap.sh --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,regex,serialization,system,thread --prefix=$HOME/lib/boost``
  - ``./b2 cxxflags="-std=c++11" -j4 && ./b2 install``
- *environment:* (assumes install from source in ``$HOME/lib/boost``)

  - ``export BOOST_ROOT=$HOME/lib/boost``
  - ``export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH``

git
"""
- 1.7.9.5 or `higher <https://help.github.com/articles/https-cloning-errors>`_
- *Debian/Ubuntu:* ``sudo apt-get install git``
- *Arch Linux:* ``sudo pacman --sync git``
- *Spack:* ``spack install git``

rsync
"""""
- *Debian/Ubuntu:* ``sudo apt-get install rsync``
- *Arch Linux:* ``sudo pacman --sync rsync``
- *Spack:* ``spack install rsync``

.. _install-dependencies-picongpu:

PIConGPU Source Code
^^^^^^^^^^^^^^^^^^^^

- ``git clone https://github.com/ComputationalRadiationPhysics/picongpu.git $HOME/src/picongpu``

  - *optional:* update the source code with ``cd $HOME/src/picongpu && git fetch && git pull``
  - *optional:* change to a different branch with ``git branch`` (show) and ``git checkout <BranchName>`` (switch)
- *environment*:

  - ``export PICSRC=$PICHOME/src/picongpu``
  - ``export PIC_EXAMPLES=$PICSRC/share/picongpu/examples``
  - ``export PATH=$PICSRC:$PATH``
  - ``export PATH=$PICSRC/src/tools/bin:$PATH``
  - ``export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH``

Optional Libraries
^^^^^^^^^^^^^^^^^^

CUDA
""""
- `8.0+ <https://developer.nvidia.com/cuda-downloads>`_
- required if you want to run on Nvidia GPUs
- *Debian/Ubuntu:* ``sudo apt-get install nvidia-cuda-toolkit``
- *Arch Linux:* ``sudo pacman --sync cuda``
- *Spack:* ``spack install cuda``
- at least one **CUDA** capable **GPU**
- *compute capability*: ``sm_20`` or higher (for CUDA 9+: ``sm_30`` or higher)
- `full list <https://developer.nvidia.com/cuda-gpus>`_ of CUDA GPUs and their *compute capability*
- `More <http://www.olcf.ornl.gov/titan/>`_ is always `better <http://www.cscs.ch/computers/piz_daint/index.html>`_. Especially, if we are talking GPUs :-)
- *environment:*

  - ``export CUDA_ROOT=<CUDA_INSTALL>``

If you do not install the following libraries, you will not have the full amount of PIConGPU plugins.
We recommend to install at least **pngwriter** and either **libSplash** (+ **HDF5**) or **ADIOS**.

pngwriter
"""""""""
- 0.7.0+
- *Spack:* ``spack install pngwriter``
- *from source:*

  - download from `github.com/pngwriter/pngwriter <https://github.com/pngwriter/pngwriter>`_
  - Requires `libpng <http://www.libpng.org>`_

    - *Debian/Ubuntu:* ``sudo apt-get install libpng-dev``
    - *Arch Linux:* ``sudo pacman --sync libpng``
  - example:

    - ``mkdir -p ~/src ~/build ~/lib``
    - ``git clone https://github.com/pngwriter/pngwriter.git ~/src/pngwriter/``
    - ``cd ~/build``
    - ``cmake -DCMAKE_INSTALL_PREFIX=$HOME/lib/pngwriter ~/src/pngwriter``
    - ``make install``

  - *environment:* (assumes install from source in ``$HOME/lib/pngwriter``)

    - ``export CMAKE_PREFIX_PATH=$HOME/lib/pngwriter:$CMAKE_PREFIX_PATH``
    - ``export LD_LIBRARY_PATH=$HOME/lib/pngwriter/lib:$LD_LIBRARY_PATH``

libSplash
"""""""""
- 1.7.0+ (requires *HDF5*, *boost program-options*)
- *Debian/Ubuntu dependencies:* ``sudo apt-get install libhdf5-openmpi-dev libboost-program-options-dev``
- *Arch Linux dependencies:* ``sudo pacman --sync hdf5-openmpi boost``
- *Spack:* ``spack install libsplash ^hdf5~fortran``
- *from source:*

  - ``mkdir -p ~/src ~/build ~/lib``
  - ``git clone https://github.com/ComputationalRadiationPhysics/libSplash.git ~/src/splash/``
  - ``cd ~/build``
  - ``cmake -DCMAKE_INSTALL_PREFIX=$HOME/lib/splash -DSplash_USE_MPI=ON -DSplash_USE_PARALLE=ON ~/src/splash``
  - ``make install``

- *environment:* (assumes install from source in ``$HOME/lib/splash``)

  - ``export CMAKE_PREFIX_PATH=$HOME/lib/splash:$CMAKE_PREFIX_PATH``
  - ``export LD_LIBRARY_PATH=$HOME/lib/splash/lib:$LD_LIBRARY_PATH``

HDF5
""""
- 1.8.6+
- standard shared version (no c++, enable parallel), e.g. ``hdf5/1.8.5-threadsafe``
- *Debian/Ubuntu:* ``sudo apt-get install libhdf5-openmpi-dev``
- *Arch Linux:* ``sudo pacman --sync hdf5-openmpi``
- *Spack:* ``spack install hdf5~fortran``
- *from source:*

  - ``mkdir -p ~/src ~/build ~/lib``
  - ``cd ~/src``
  - download hdf5 source code from `release list of the HDF5 group <https://www.hdfgroup.org/ftp/HDF5/releases/>`_, for example:

  - ``wget https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.14/src/hdf5-1.8.14.tar.gz``
  - ``tar -xvzf hdf5-1.8.14.tar.gz``
  - ``cd hdf5-1.8.14``
  - ``./configure --enable-parallel --enable-shared --prefix $HOME/lib/hdf5/``
  - ``make``
  - *optional:* ``make test``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/hdf5``)

  - ``export HDF5_ROOT=$HOME/lib/hdf5``
  - ``export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH``

splash2txt
""""""""""
- requires *libSplash* and *boost* ``program_options``, ``regex``
- converts slices in dumped hdf5 files to plain txt matrices
- assume you [downloaded](#requirements) PIConGPU to `PICSRC=$HOME/src/picongpu`
- ``mkdir -p ~/build && cd ~/build``
- ``cmake -DCMAKE_INSTALL_PREFIX=$PICSRC/src/tools/bin $PICSRC/src/tools/splash2txt``
- ``make``
- ``make install``
- *environment:*

  - ``export PATH=$PATH:$PICSRC/src/splash2txt/build``
- options:

  - ``splash2txt --help``
  - list all available datasets: ``splash2txt --list <FILE_PREFIX>``

png2gas
"""""""
- requires *libSplash*, *pngwriter* and *boost* ``program_options``)
- converts png files to hdf5 files that can be used as an input for a species initial density profiles
- compile and install exactly as *splash2txt* above

ADIOS
"""""
- 1.10.0+ (requires *MPI* and *zlib*)
- *Debian/Ubuntu:* ``sudo apt-get install libadios-dev libadios-bin``
- *Arch Linux* using an `AUR helper <https://wiki.archlinux.org/index.php/AUR_helpers>`_: ``pacaur --sync libadios``
- *Arch Linux* using the `AUR <https://wiki.archlinux.org/index.php/Arch_User_Repository>`_ manually:

  - ``sudo pacman --sync --needed base-devel``
  - ``git clone https://aur.archlinux.org/libadios.git``
  - ``cd libadios``
  - ``makepkg -sri``
- *Spack:* ``spack install adios``
- *from source:*

  - ``mkdir -p ~/src ~/build ~/lib``
  - ``cd ~/src``
  - ``wget http://users.nccs.gov/~pnorbert/adios-1.10.0.tar.gz``
  - ``tar -xvzf adios-1.10.0.tar.gz``
  - ``cd adios-1.10.0``
  - ``CFLAGS="-fPIC" ./configure --enable-static --enable-shared --prefix=$HOME/lib/adios --with-mpi=$MPI_ROOT --with-zlib=/usr``
  - ``make``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/adios``)

  - ``export ADIOS_ROOT=$HOME/lib/adios``
  - ``export LD_LIBRARY_PATH=$ADIOS_ROOT/lib:$LD_LIBRARY_PATH``

ISAAC
"""""
- 1.3.0+
- requires *boost* (header only), *IceT*, *Jansson*, *libjpeg* (preferably *libjpeg-turbo*), *libwebsockets* (only for the ISAAC server, but not the plugin itself)
- enables live in situ visualization, see more here `Plugin description <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/Plugin%3A-ISAAC>`_
- *Spack:* ``spack install isaac``
- *from source:* build the *in situ library* and its dependencies as described in `ISAAC's INSTALL.md <https://github.com/ComputationalRadiationPhysics/isaac/blob/master/INSTALL.md>`_
- *environment:* set environment variable ``CMAKE_PREFIX_PATH`` for each dependency and the ISAAC in situ library

VampirTrace
"""""""""""
- for developers: performance tracing support
- download 5.14.4 or higher, e.g. from `www.tu-dresden.de <https://tu-dresden.de/zih/forschung/projekte/vampirtrace>`_
- *from source:*

  - ``mkdir -p ~/src ~/build ~/lib``
  - ``cd ~/src``
  - ``wget -O VampirTrace-5.14.4.tar.gz "http://wwwpub.zih.tu-dresden.de/~mlieber/dcount/dcount.php?package=vampirtrace&get=VampirTrace-5.14.4.tar.gz"``
  - ``tar -xvzf VampirTrace-5.14.4.tar.gz``
  - ``cd VampirTrace-5.14.4``
  - ``./configure --prefix=$HOME/lib/vampirtrace --with-cuda-dir=<CUDA_ROOT>``
  - ``make all -j``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/vampirtrace``)

  - ``export VT_ROOT=$HOME/lib/vampirtrace``
  - ``export PATH=$VT_ROOT/bin:$PATH``
