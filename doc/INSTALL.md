PIConGPU Install Guide
======================

Requirements
------------

### Mandatory

- g++ 4.3 to 4.6 (depends on current CUDA version)
- CUDA Toolkit 5.5 or higher (or >=4.2 for <`sm_3x` cards)
- nvidia cards: `sm_13` for basic PIC cycle, >=`sm_20` for features (higher order shapes, radiation, ...)
- cmake 2.6 or higher
- OpenMPI 1.4 or higher
- zlib (tested with 1.2.7 or higher)
- boost 1.47.0 or higher ("program options", "regex" and nearly all compile time libs)
    - download from [http://www.boost.org/](http://sourceforge.net/projects/boost/files/boost/1.49.0/boost_1_49_0.tar.gz/download),
      e.g. version 1.49.0
- subversion 1.6 or higher to get the sources 
    - (subversion 1.5.7 seems to be possible but is not recommended)
- PIConGPU
    - `$ svn co --username <USER> https://fusionforge.zih.tu-dresden.de/svn/picongpu2/ <PIC_DIR>`

### Optional Libraries

If you do not install the optional libraries, you will not have the full amount of PIConGPU online analysers.
We recomment to install at least "pngwriter".

- pngwriter
    - download from
      [http://pngwriter.sourceforge.net/](http://sourceforge.net/projects/pngwriter/files/pngwriter/pngwriter-0.5.4/pngwriter-0.5.4.tar.gz/download),
      e.g. version 0.5.4
    - BUG in v0.5.4: please change in examples/pngtest.cc:48 from `#include <iostream.h>` to `#include <iostream>`

- libSplash (requires hdf5)
    - `$ svn co --username <USER> https://fusionforge.zih.tu-dresden.de/svn/datacollector/trunk/ <SPLASH_ROOT_DIR>`
    - create a build `<BUILD>`  and a splash directory `<SPLASH_INSTALL>`
    - build in `<BUILD>` directory:
        - `$ cmake -DCMAKE_INSTALL_PREFIX=<SPLASH_INSTALL> <SPLASH_ROOT_DIR>`
        - `$ make`
        - `$ make install` creates splash library in `<SPLASH_INSTALL>`
    - for environment variables see: "Additional required environment variables"

- hdf5 1.8.x, standard shared version (no c++, not parallel), e.g. hdf5/1.8.5-threadsafe
    - configure example:
      `$ ./configure --enable-threadsafe --prefix $HOME/lib/hdf5/ --with-pthread=/lib`

- splash2txt (libSplash and boost "program_options", "regex" required)
    - inofficial tool shipped with the PIConGPU
    - `$ cd src/splash2txt/build`
    - `$ cmake ..`
    - `$ make`
    - options: `$ splash2txt --help`
    - list all available datasets: `$ splash2txt --list <FILE_PREFIX>`

- for VampirTrace support
    - download 5.14.4 or higher, e.g. from 
    [http://www.tu-dresden.de/die_tu_dresden/zentrale_einrichtungen/zih/forschung/projekte/vampirtrace](http://www.tu-dresden.de/die_tu_dresden/zentrale_einrichtungen/zih/forschung/projekte/vampirtrace)
    - build VampirTrace:
        - extract with `$ tar -xfz VampirTrace-5.14.4.tar.gz`
        - `$ ./configure --prefix=<VT_DIR> --with-cuda-dir=<CUDA_ROOT>`
        - `$ make all -j`
        - `$ make install`
        - add environment variable
          `$ export PATH=$PATH:<VT_DIR>/bin`

*******************************************************************************


Install
-------

### Mandatory environment variables

- CUDA\_LIB: library directory of cuda: 
    e.g. `$ export CUDA_LIB=<CUDA_INSTALL>/lib64`
- MPI\_ROOT: mpi installation directory: 
    e.g. `$ export MPI_ROOT=<MPI_INSTALL>`
- extend your PATH with helper tools for PIConGPU, see point:
    [Checkout and Build PIConGPU](#checkout-and-build-picongpu)


### Additional required environment variables (for optional libraries)

#### for splash and HDF5
- SPLASH\_ROOT: libsplash installation directory: 
    e.g. `$ export SPLASH_ROOT=<SPLASH_INSTALL>`
- HDF5\_ROOT: hdf5 installation directory: 
    e.g. `$ export HDF5_ROOT=<HDF5_INSTALL>`
- LD\_LIBRARY\_PATH: add path to SPLASH/lib: 
    e.g. `$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<SPLASH_INSTALL>/lib`

#### for png support
- PNGWRITER\_ROOT: pngwriter installation directory:
  e.g. `$ export PNGWRITER_ROOT=<PNGWRITER_INSTALL>`

#### environment variables for tracing
- VT\_ROOT: VampirTrace installation directory:
    e.g. `$ export PATH=$PATH:<VT_DIR>`

### Installation notes
- Be sure to build all libraries/dependencies with the **same** gcc version.
- Never set a environment variable to the source folder, always set them to the
  installation directory.

*******************************************************************************


Checkout and Build PIConGPU
---------------------------

This is an example how to use the modular building environment of PIConGPU.

1. `$ mkdir picongpu_workspace`
   - create a folder on your hard disk and go there
   - `$ cd picongpu_workspace`
2. Create the following directory structure:
   1. `$ mkdir svn_code`
      - create a folder for the PIConGPU source code, it is used as *read only*
        directory
   2. `$ mkdir build`
      - temporary directory for build processes
   3. `$ mkdir paramSets`
      - stores different parameter sets
   4. `$ mkdir runs`
      - directory for PIConGPU runtime simulation output
      - NOTE for HPC-Systems: Never write your simulation output to your home
        directory (in most cases `$WORK` or `/scratch` are the right places).
3. Download the source code:
   1. `$ svn co --username <USER> https://fusionforge.zih.tu-dresden.de/svn/picongpu2/trunk svn_code`
      - *optional:* update the source code with `$ svn update svn_code/`
   2. `$ export PATH=$PATH:<ABSOLUTE_PATH>/svn_code/src/tools/bin`
      - ATTENTION: replace `<ABSOLUTE_PATH>` with the absolute path to the
        directory `svn_code` (try `$ pwd` to get your current path)
4. `$ svn_code/createParameterSet code/examples/LaserWakefield/ paramSets/case001`
   - Clone the KHI example tp `paramSets/case001`
   - Edit `paramSets/case001/include/simulation_defines/param/*` to change the
     physical configuration of this parameter set.
   - See `$ svn_code/createParameterSet --help` for more options.
   - *optional:* `$ svn_code/createParameterSet paramSets/case001/ paramSets/case002`
   - Clone the individual parameterSet `case001` to `case002`.
   - *optional:* `$ svn_code/createParameterSet paramSets/myset`
   - Create standard parameter files and copy the set to `myset`.
5. `$ cd build`
   - go to the build directory to compile your first test
6. `$ ../svn_code/configure ../paramSets/case001`
    - *optional:* `$ ../svn_code/configure --help`
    - NOTE: *makefiles* are always created in the current directory
    - Configure *one* parameter set for *one* compilation
    - The script `configure` is only a wrapper for cmake. The real `cmake`
      command will be shown in the first line of the output.
    - `case001` is the directory were the full compiled binary with all
      parameter files will be installed to
7. `$ make [-j]`
    - compile PIConGPU with the current parameter set: `case001`
8. `$ make install`
    - copy binaries and params to a fixed data structure: `case001`
9. `cd ../paramSets/case001`
    - goto installed programm
10. Example run for the HPC System "joker" using a batch system
    - *optional:* `$ tbg --help`
    - `$ tbg -s qsub -c submit/joker/picongpu.cfg
             -t submit/joker/picongpu.tpl ../../runs/testBatch01`
    - This will create the directory `../../runs/testBatch01` were all
      simulation output will be written to. 
      This folder has a subfolder `picongpu` with the same structure as
      `case001` and can be reused to:
        - clone parameters as shown in step 9, by using this run as origin
        - create a new binary with configure (step 14):
          e.g. `$ <PATH_TO_SVN>/configure -i paramSets/case002 runs/testBatch01`


To build PIConGPU with tracing support, change the steps in the example to:

(6.) `$ ../svn_code/configure ../projects/case001 -c "-DVAMPIR_ENABLE=ON"`

(9.) `$ cd ../projects/case001` - goto installed programm

(10.) `$ tbg -c submit/joker/vampir.cfg -t submit/joker/vampir.tpl  ../../runs/testBatch01`

*******************************************************************************


Notes:
------

This document uses markdown syntax: http://daringfireball.net/projects/markdown/

To create and up-to-date pdf out of this Markdown (.md) file use gimli.
Anyway, please do *not* check in the binary pdf in our version control system!

  - `$ sudo apt-get install rubygems wkhtmltopdf libxslt-dev libxml2-dev`
  - `$ sudo gem install gimli`
  - `$ gimli -f INSTALL.md`

On OS X (Apple Macintosh) you can get `gimli` using the already installed
`ruby-gems` (`gem`) or install a newer version using macports:

1. *install macports: http://www.macports.org/*
2. *install rubygems* <br>
    `$ sudo port install rb-rubygems ` <br>
    *rubygems is called gem by default*
3. install gimli <br>
    `$ sudo gem install gimli`<br>
    (installs all libraries automatically)
4. convert documentation to pdf <br>
    `$ gimli -f INSTALL.md`
