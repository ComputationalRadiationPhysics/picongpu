cupla Install Guide
======================

Requirements
------------

- **cmake**  3.11.0 or higher
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman --sync cmake`

- **cupla**
  - https://github.com/ComputationalRadiationPhysics/cupla
  - `export CUPLA_ROOT=<cupla_SRC_CODE_DIR>`
  - `export CMAKE_PREFIX_PATH=$CUPLA_ROOT:$CMAKE_PREFIX_PATH`
  - example:
    - `mkdir -p $HOME/src`
    - `git clone git://github.com/ComputationalRadiationPhysics/cupla.git $HOME/src/cupla`
    - `cd $HOME/src/cupla`
    - `export CUPLA_ROOT=$HOME/src/cupla`
  - use a different alpaka installation:
    set environment variable `ALPAKA_ROOT` or extend `CMAKE_PREFIX_PATH` with the
    path to alpaka.


Compile an example
------------------

- create build directory `mkdir -p buildCuplaExample`
- `cd buildCuplaExample`
- `cmake $CUPLA_ROOT/example/CUDASamples/matrixMul -D<ACC_TYPE>=ON`
    - list of supported `ACC_TYPE`s
        - `ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE`
        - `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE`
        - `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE`
        - `ALPAKA_ACC_GPU_CUDA_ENABLE`
        - `ALPAKA_ACC_CPU_BT_OMP4_ENABLE`
        - `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE` (only allowed in combination with
          `CUPLA_KERNEL_OPTI` and `CUPLA_KERNEL_ELEM`, because the `blockSize` must be `dim3(1,1,1)`)
          see [TuningGuide.md](doc/TuningGuide.md)
- `make -j`
- `./matrixMul -wA=320 -wB=320 -hA=320 -hB=320` (parameters must be a multiple of 32!)


How to update alpaka as git subtree?
------------------------------------

```zsh
# git author is generic to not mess up contribution statistics
GIT_AUTHOR_NAME="Third Party" GIT_AUTHOR_EMAIL="crp-git@hzdr.de" \
 git subtree pull --prefix alpaka \
 https://github.com/ComputationalRadiationPhysics/alpaka.git develop --squash
```

**How to commit local changes to alpaka upstream?**

If your local alpaka version contains changes you want to contribute back upstream via fork, then you can use `git subtree push`:

```zsh
# Add your fork of alpaka to git remotes
git remote add alpaka-fork git@github.com:YOUR_NAME/alpaka.git

# Push your changes to your fork
git subtree push --prefix=alpaka alpaka-fork
```
Then check your github page of your fork to open a pull request upstream.

More information can be found in this [git subtree guide](https://www.atlassian.com/blog/git/alternatives-to-git-submodule-git-subtree).
