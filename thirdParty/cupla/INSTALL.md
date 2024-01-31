# cupla Install Guide

## Requirements


- **cmake**  3.22.0 or higher
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman --sync cmake`

- **boost** 1.74.0 or higher
  - is required by alpaka
  - *Install Instructions:* https://github.com/alpaka-group/alpaka#dependencies

## Compile an example

```bash
cd <path/to/cupla>
mkdir build && cd build
cmake .. -Dcupla_BUILD_EXAMPLES=ON -D<ACC_TYPE>=ON
cmake --build .
# execute an example
# (parameters must be a multiple of 32!)
example/CUDASamples/matrixMul/matrixMul -wA=320 -wB=320 -hA=320 -hB=320
```

- list of supported `ACC_TYPE`s
  - `alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE`
  - `alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE`
  - `alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE`
  - `alpaka_ACC_GPU_CUDA_ENABLE`
  - `alpaka_ACC_GPU_HIP_ENABLE`
  - `alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE` (only allowed in combination with
  `CUPLA_KERNEL_OPTI` and `CUPLA_KERNEL_ELEM`, because the `blockSize` must be `dim3(1,1,1)`)
  see [TuningGuide.md](doc/TuningGuide.md)

The CMake argument `cupla_ALPAKA_PROVIDER` controls which `alpaka` is used. If the value is `internal` (default), the alpaka, ship with `cupla` is used. If the value is `external`, am installed `aplaka` via `find_package()` is used.

## Use cupla in your Project

`cupla` offers two ways to use it in your project via CMake.

1. You can install it and use it via `find_package()`
2. You can add the `cupla` project to your project and use it with `add_subdirectory()`.

### Install Cupla (find_package method)

The installation of `cupla` requires an installed version of `alpaka`. You can install the `alpaka` version in the `cupla` directory (recommended) or install `aplaka` directly from [source](https://github.com/alpaka-group/alpaka). **Attention**, there is no warning if you overwrite an existing `alpaka` installation.


```bash
cd <path/to/alpaka>
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

`cupla` can be installed via

```
cd <path/to/cupla>
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

To check your installation, you can activate the examples (see section [compiling an example](https://github.com/alpaka-group/cupla/blob/dev/INSTALL.md#compile-an-example)) during the configuration of `cupla` and check whether it compiles. Activating the examples does not change anything in the installed `cupla` library.

```bash
# use external alpka to verify your environment
cmake .. -Dcupla_BUILD_EXAMPLES=ON -Dcupla_ALPAKA_PROVIDER="external"
```

If cupla is installed, you can use it via `find_package()` in your project:

```cmake
cmake_minimum_required(VERSION 3.22.0)
project(exampleProject)

find_package(cupla)

cupla_add_executable(${PROJECT_NAME} main.cpp)
```

### add_subdirectory

For the method `add_subdirectory()` you have to copy the `cupla` project folder into your project. After that you can add `cupla` to your project and use it.

```cmake
cmake_minimum_required(VERSION 3.22.0)
project(exampleProject)

# requires, that cupla is located in the root directory of your project
add_subdirectory(cupla)

cupla_add_executable(${PROJECT_NAME} main.cpp)
```

Via the CMake variable `cupla_ALPAKA_PROVIDER` you can decide whether you want to use the `alpaka` ship with `cupla` (`-Dcupla_ALPAKA_PROVIDER="interal"`) or an installed version of `alpaka` (`-Dcupla_ALPAKA_PROVIDER="external"` - uses `find_package(alpaka)`).

## How to update alpaka as git subtree?

```zsh
# git author is generic to not mess up contribution statistics
GIT_AUTHOR_NAME="Third Party" GIT_AUTHOR_EMAIL="crp-git@hzdr.de" \
 git subtree pull --prefix alpaka \
 https://github.com/alpaka-group/alpaka.git develop --squash
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
