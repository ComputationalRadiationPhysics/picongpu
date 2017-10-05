cupla Install Guide
======================

Requirements
------------

- **cmake**  3.3.0 or higher
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman --sync cmake`

- **cupla**
  - https://github.com/ComputationalRadiationPhysics/cupla
  - `export CUPLA_ROOT=<cupla_SRC_CODE_DIR>`
  - example:
    - `mkdir -p $HOME/src`
    - `git clone git://github.com/ComputationalRadiationPhysics/cupla.git $HOME/src/cupla`
    - `cd $HOME/src/cupla`
    - `git submodule init`
    - `git submodule update`
    - `export CUPLA_ROOT=$HOME/src/cupla`
  - use cupla without the submodule alpaka: 
    Set the advanced CMake variable `cupla_ALPAKA_PROVIDER` to `extern` and
    set environment variable `ALPAKA_ROOT` or extend `CMAKE_PREFIX_PATH` with the
    path to alpaka.
      

compile an example
-----------------

- create build directory `mkdir -p buildCuplaExample`
- `cd buildCuplaExample`
- `cmake $CUPLA_ROOT/example/CUDASamples/matrixMul -D<ACC_TYPE>=ON`
    - list of supported ACC_TYPES
        - `ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE`
        - `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE`
        - `ALPAKA_ACC_GPU_CUDA_ENABLE`
        - `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE` (only allowed in combination with
          `CUPLA_KERNEL_OPTI` and `CUPLA_KERNEL_ELEM`, because the `blockSize` must be dim3(1,1,1))
          see [TuningGuide.md](doc/TuningGuide.md)
- `make -j`
- `./matrixMul -wA=320 -wB=320 -hA=320 -hB=320` (parameters must be a multiple of 32!)
