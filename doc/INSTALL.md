PIConGPU Install Guide
======================

Requirements
------------

### Mandatory

- **gcc** 4.3 to 4.6 (depends on current CUDA version)
  - *Debian/Ubuntu:* `apt-get install gcc-4.4 build-essential`

- [CUDA 5.5](https://developer.nvidia.com/cuda-downloads) or higher

- at least one **CUDA** capable **GPU**
  - **sm\_13** for basic PIC cycle, >= **sm\_20** for all features (higher order shapes, radiation diagnostics, ...)
  - [full list](https://developer.nvidia.com/cuda-gpus) of CUDA GPUs and their *compute capability*
  - ([More](http://www.olcf.ornl.gov/titan/) is always better. Especially, if we are talking about GPUs.)

- **cmake** 2.8 or higher
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`

- **OpenMPI** 1.4 or higher
  - Debian/Ubuntu: `sudo apt-get install libopenmpi-dev`

- **zlib**
  - *Debian/Ubuntu:* `sudo apt-get install zlib1g-dev`

- **boost** 1.47.0 or higher ("program options", "regex" and nearly all compile time libs)
  - download from [http://www.boost.org/](http://sourceforge.net/projects/boost/files/boost/1.49.0/boost_1_49_0.tar.gz/download),
      e.g. version 1.49.0
  - *Debian/Ubuntu:* `sudo apt-get install libboost-program-options-dev libboost-regex-dev`

- **git**
  - *Debian/Ubuntu:* `sudo apt-get install git`

- **PIConGPU**
    - `git clone https://github.com/ComputationalRadiationPhysics/picongpu.git ~/src/picongpu`

### Optional Libraries

If you do not install the optional libraries, you will not have the full amount of PIConGPU online analysers.
We recomment to install at least **pngwriter**.

- **pngwriter**
    - download our modified version from
      [github.com/ax3l/pngwriter](https://github.com/ax3l/pngwriter)
    - Requires [libpng](http://www.libpng.org/), *Debian/Ubuntu:* `sudo apt-get install libpng-dev`
    - example:
      - `mkdir -p ~/src ~/build ~/lib`
      - `git clone https://github.com/ax3l/pngwriter.git ~/src/pngwriter/`
      - `cd ~/build`
      - `cmake -DCMAKE_INSTALL_PREFIX=~/lib/pngwriter ~/src/pngwriter`
      - `make install`
    - set the environment variable
      [PNGWRITER\_ROOT](#additional-required-environment-variables-for-optional-libraries)
      to `$HOME/lib/pngwriter`

- **libSplash** (requires *hdf5*)
    - example:
      - `mkdir -p ~/src ~/build ~/lib`
      - `git clone https://github.com/ComputationalRadiationPhysics/libSplash.git ~/src/splash/`
      - `cd ~/build`
      - `cmake -DCMAKE_INSTALL_PREFIX=~/lib/splash ~/src/splash`
      - `make install`
    - set the environment variable
      [SPLASH\_ROOT](#additional-required-environment-variables-for-optional-libraries)
      to `$HOME/lib/splash`

- **hdf5** >= 1.8.6, standard shared version (no c++)
    - *Debian/Ubuntu:* `sudo apt-get install libhdf5-dev`
    - example:
      - `mkdir -p ~/src ~/build ~/lib`
      - `cd ~/src`
      - `wget www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.11.tar.gz`
      - `tar -xvzf hdf5-1.8.11.tar.gz`
      - `cd hdf5-1.8.11`
      - `./configure --enable-shared --prefix ~/lib/hdf5/`
      - `make`
      - *optional:* `make test`
      - `make install`
    - set the environment variable
      [HDF5\_ROOT](#additional-required-environment-variables-for-optional-libraries)
      to `$HOME/lib/hdf5` and the
      [LD\_LIBRARY\_PATH](#additional-required-environment-variables-for-optional-libraries)
      to `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib/hdf5/lib`

- **splash2txt** (requires *libSplash* and *boost* "program_options", "regex")
    - inofficial tool shipped with the PIConGPU
    - assume you [downloaded](#requirements) PIConGPU to `~/src/picongpu`
    - `mkdir -p ~/build && cd ~/build`
    - `cmake ~/src/picongpu/src/splash2txt`
    - `make`
    - copy the binary `splash2txt` to a place in your `$PATH`)
    - options:
      - `splash2txt --help`
      - list all available datasets: `splash2txt --list <FILE_PREFIX>`

- for **VampirTrace** support
    - download 5.14.4 or higher, e.g. from 
    [http://www.tu-dresden.de](http://www.tu-dresden.de/die_tu_dresden/zentrale_einrichtungen/zih/forschung/projekte/vampirtrace)
    - example:
      - `mkdir -p ~/src ~/build ~/lib`
      - `cd ~/src`
      - `wget -O VampirTrace-5.14.4.tar.gz "http://wwwpub.zih.tu-dresden.de/~mlieber/dcount/dcount.php?package=vampirtrace&get=VampirTrace-5.14.4.tar.gz"`
      - `tar -xvzf VampirTrace-5.14.4.tar.gz`
      - `cd VampirTrace-5.14.4`
      - `./configure --prefix=$HOME/lib/vampirtrace --with-cuda-dir=<CUDA_ROOT>`
      - `make all -j`
      - `make install`
    - add environment variable: `export PATH=$PATH:$HOME/lib/vampirtrace/bin`

*******************************************************************************


Install
-------

### Mandatory environment variables

- `CUDA_ROOT`: cuda installation directory, 
    e.g. `export CUDA_ROOT=<CUDA_INSTALL>`
- `MPI_ROOT`: mpi installation directory, 
    e.g. `export MPI_ROOT=<MPI_INSTALL>`
- extend your `$PATH` with helper tools for PIConGPU, see point,
    [Checkout and Build PIConGPU](#checkout-and-build-picongpu) *step 2.2*


### Additional required environment variables (for optional libraries)

#### for splash and HDF5
- `SPLASH_ROOT`: libsplash installation directory, 
    e.g. `export SPLASH_ROOT=$HOME/lib/splash`
- `HDF5_ROOT`: hdf5 installation directory, 
    e.g. `export HDF5_ROOT=$HOME/lib/hdf5`
- `LD_LIBRARY_PATH`: add path to $HOME/lib/hdf5/lib,
    e.g. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib/hdf5/lib`

#### for png support
- `PNGWRITER_ROOT`: pngwriter installation directory,
  e.g. `export PNGWRITER_ROOT=<PNGWRITER_INSTALL>`

#### environment variables for tracing
- `VT_ROOT`: VampirTrace installation directory,
    e.g. `export PATH=$PATH:$HOME/lib/vampirtrace/bin`

### Installation notes
- Be sure to build all libraries/dependencies with the **same** gcc version.
- Never set a environment variable to the source folder, always set them to the
  installation directory.

*******************************************************************************


Checkout and Build PIConGPU
---------------------------

This is an example how to use the modular building environment of PIConGPU.

1. **Setup directories:** `mkdir -p ~/src ~/build ~/paramSets ~/runs`
   - `~/runs` is the directory for PIConGPU simulation output
   - NOTE for HPC-Systems: Never write your simulation output to your home
     (=`~/`) directory
   - In most cases `$WORK/runs`, `$WORKDIR/runs` or `/scratch/runs` are the right places!
2. **Download the source code:**
   1. `git clone https://github.com/ComputationalRadiationPhysics/picongpu.git ~/src/picongpu`
      - *optional:* update the source code with `cd ~/src/picongpu && git pull`
      - *optional:* change to a different branch with `git branch` (show) and
                    `git checkout <BranchName>` (switch)
   2. `export PATH=$PATH:$HOME/src/picongpu/src/tools/bin`
   3. `export PICSRC=$HOME/src/picongpu`
3. **Create parameterset:** `$PICSRC/createParameterSet $PICSRC/examples/LaserWakefield/ ~/paramSets/case001`
   - Clone the LWFA example to `~/paramSets/case001`
   - Edit `~/paramSets/case001/include/simulation_defines/param/*` to change the
     physical configuration of this parameter set.
   - See `$PICSRC/createParameterSet --help` for more options.
   - *optional:* `$PICSRC/createParameterSet ~/paramSets/case001/ ~/paramSets/case002`
     - Clone the individual parameterSet `case001` to `case002`.
4. `cd ~/build`: go to the build directory to compile your first test
5. **Configure:** `$PICSRC/configure ~/paramSets/case001`
    - *optional:* `$PICSRC/configure --help`
    - NOTE: *makefiles* are always created in the current directory
    - Configure *one* parameter set for *one* compilation
    - The script `configure` is only a wrapper for cmake. The real `cmake`
      command will be shown in the first line of the output.
    - `case001` is the directory were the full compiled binary with all
      parameter files will be installed to
6. **Compile:** `make [-j]`: compile PIConGPU with the current parameter set: `case001`
7. **Install:** `make install`: copy binaries and params to a fixed data structure: `case001`
8. `cd ~/paramSets/case001`: goto installed programm
9. **Run:** Example run for the HPC System "hypnos" using a PBS batch system
    - *optional:* `tbg --help`
    - `tbg -s qsub -c submit/0016.gpus.cfg
           -t submit/hypnos/k20_profile.tpl ~/runs/testBatch01`
    - This will create the directory `~/runs/testBatch01` were all
      simulation output will be written to.
      ( *Again*, do NOT use your home `~/runs` - change this path!)
    - `tbg` will create a subfolder `picongpu` in the directory of the run with
      the same structure as `case001`. It can be reused to:
        - clone parameters as shown in step 3, by using this run as origin
        - create a new binary with configure (step 5):
          e.g. `$PICSRC/configure -i ~/paramSets/case002 ~/runs/testBatch01`


To build PIConGPU with tracing support, change the steps in the example to:

(5.) `$PICSRC/configure ../projects/case001 -c "-DVAMPIR_ENABLE=ON"`

(8.) `cd ~/paramSets/case001`: goto installed programm

(9.) `tbg -c submit/0016.gpus.cfg -t submit/hypnos/k20_vampir_profile.tpl
       ~/runs/testBatchVampir`

*******************************************************************************


Notes:
------

This document uses markdown syntax: http://daringfireball.net/projects/markdown/

To create and up-to-date pdf out of this Markdown (.md) file use gimli.
Anyway, please do *not* check in the binary pdf in our version control system!

  - `sudo apt-get install rubygems wkhtmltopdf libxslt-dev libxml2-dev`
  - `sudo gem install gimli`
  - `gimli -f INSTALL.md`

On OS X (Apple Macintosh) you can get `gimli` using the already installed
`ruby-gems` (`gem`) or install a newer version using macports:

1. *install macports: http://www.macports.org/*
2. *install rubygems* <br>
    `sudo port install rb-rubygems ` <br>
    *rubygems is called gem by default*
3. install gimli <br>
    `sudo gem install gimli`<br>
    (installs all libraries automatically)
4. convert documentation to pdf <br>
    `gimli -f INSTALL.md`
