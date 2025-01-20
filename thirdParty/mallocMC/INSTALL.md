Install
-------
### Dependencies
 - C++20 compiler (clang, gcc, hipcc, icc, nvcc)
  - *Debian/Ubuntu:* `sudo apt-get install gcc build-essential`
  - *Arch Linux:* `sudo pacman -S base-devel`
 - `alpaka` 1.2.0
  - included as git submodule
 - `boost` >= 1.65.1
   - dependency of alpaka
   - *Debian/Ubuntu:* `sudo apt-get install libboost-dev libboost-program-options-dev`
   - *Arch Linux:* `sudo pacman -S boost`
   - or download from [http://www.boost.org/](http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download)
 - `CMake` >= 3.15
  - *Debian/Ubuntu:* `sudo apt-get install cmake file cmake-curses-gui`
  - *Arch Linux:* `sudo pacman -S cmake`
 - `git` >= 1.7.9.5
  - *Debian/Ubuntu:* `sudo apt-get install git`
  - *Arch Linux:* `sudo pacman -S git`


### Examples
This is an example how to compile `mallocMC` and test the example code snippets

1. **Setup directories:**
 - `mkdir -p build`
2. **Download the source code:**
 -  `git clone https://github.com/alpaka-group/mallocMC.git`
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
- `-lboost`

If you are using CMake you can download our `FindmallocMC.cmake` module with
```bash
wget https://raw.githubusercontent.com/ComputationalRadiationPhysics/cmake-modules/dev/FindmallocMC.cmake
# read the documentation
cmake -DCMAKE_MODULE_PATH=. --help-module FindmallocMC | less
```

and use the following lines in your `CMakeLists.txt`:
```cmake
# this example will require at least CMake 3.15
CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# add path to FindmallocMC.cmake, e.g., in the directory in cmake/
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

# find mallocMC installation
find_package(mallocMC 2.6.0 REQUIRED)

alpaka_add_executable(yourBinary ${SOURCES})
target_include_directories(yourBinary PUBLIC ${mallocMC_INCLUDE_DIRS})
target_link_libraries(yourBinary PUBLIC alpaka::alpaka)
```
