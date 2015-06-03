Install
-------
### Dependencies
 - `gcc` 4.4 - 4.8 (depends on current CUDA version)
  - *Debian/Ubuntu:* `sudo apt-get install gcc-4.4 build-essential`
  - *Arch Linux:* `sudo pacman -S base-devel`
 - [CUDA 5.0](https://developer.nvidia.com/cuda-downloads) or higher
  - *Debian/Ubuntu:* `sudo apt-get install nvidia-common nvidia-current nvidia-cuda-toolkit nvidia-cuda-dev`
  - *Arch Linux:* `sudo pacman -S cuda`
 - one Nvidia **CUDA** compatible **GPU** with compute capability >= 2.0
  - [full list](https://developer.nvidia.com/cuda-gpus) of CUDA GPUs and their *compute capability*
 - `boost` >= 1.48
   - compile time headers
   - `boost::program_options`
   - *Debian/Ubuntu:* `sudo apt-get install libboost-dev libboost-program-options-dev`
   - *Arch Linux:* `sudo pacman -S boost`
   - or download from [http://www.boost.org/](http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download)
 - `CMake` >= 2.8.5
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman -S cmake`
 - `git` >= 1.7.9.5
  - *Debian/Ubuntu:* `sudo apt-get install git`
  - *Arch Linux:* `sudo pacman -S git`


### Mandatory environment variables
 - `CUDA_ROOT`: CUDA installation directory, e.g. `export CUDA_ROOT=<CUDA_INSTALL>`
  - this might be already set through your CUDA toolkit
 - `BOOST_ROOT`: Boost installation directory, e.g. `export BOOST_ROOT=<BOOST_INSTALL>`

### Examples
This is an example how to compile `mallocMC` and test the example code snippets

1. **Setup directories:**
 - `mkdir -p build`
2. **Download the source code:**
 -  `git clone https://github.com/ComputationalRadiationPhysics/mallocMC.git`
3. **Build**
 - `cd build`
 - `cmake ../mallocMC -DCMAKE_INSTALL_PREFIX=$HOME/libs`
 - `make examples`
 - `make install` (optional)
4. **Run the examples**
 - `./mallocMC_Example01`
 - `./mallocMC_Example02`
 - `./VerifyHeap`
  - additional options: see `./VerifyHeap --help`


Linking to your Project
-----------------------

To use mallocMC in your project, you must include the header `mallocMC/mallocMC.hpp` and
add the correct include path.

Because we are linking to Boost and CUDA, the following **external dependencies** must be linked:
- `-lboost`, `-lcudart`

If you are using CMake you can download our `FindmallocMC.cmake` module with
```bash
wget https://raw.githubusercontent.com/ComputationalRadiationPhysics/cmake-modules/dev/FindmallocMC.cmake
# read the documentation
cmake -DCMAKE_MODULE_PATH=. --help-module FindmallocMC | less
```

and use the following lines in your `CMakeLists.txt`:
```cmake
# this example will require at least CMake 2.8.5
cmake_minimum_required(VERSION 2.8.5)

# add path to FindmallocMC.cmake, e.g., in the directory in cmake/
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

# find the packages that are required by mallocMC. This has to be done BEFORE
# loading mallocMC
find_package(Boost REQUIRED)
set(LIBS ${LIBS} ${Boost_LIBRARIES})

find_package(CUDA REQUIRED)
cuda_include_directories(${CUDA_INCLUDE_DIRS})

# find mallocMC installation
find_package(mallocMC 2.0.1 REQUIRED)

# where to find headers (-I includes for compiler)
include_directories(SYSTEM ${mallocMC_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(yourBinary ${SOURCES})

target_link_libraries(yourBinary ${LIBS})
```
