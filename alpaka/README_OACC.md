# Building configuring for the OpenACC backend

## CMake Basics

Alpaka requires cmake version >= 3.18.

In the root of the alpaka dir, run:
```bash
mkdir build
cd build
```

## Configuring Using CMake

In the build directory, invoke cmake to configure. Use the options below to
enable only the OpenACC backend.

```bash
cmake .. \
  -Dalpaka_ACC_ANY_BT_OACC_ENABLE=on \
  -DBUILD_TESTING=on \
  -Dalpaka_BUILD_EXAMPLES=on \
```
All other backends are disabled for faster compilation/testing and reduced
environment requirements. The cmake package OpenACC is used to detect the
required OpenACC flags for the compiler. Additional flags can be added, e.g:
- gcc, target x86:
  ```bash
    -DCMAKE_CXX_FLAGS="-foffload=disable"
  ```
  - To run set the environment variable `ACC_DEVICE_TYPE=host`.
  - As of gcc 9.2 no test will compile if the nvptx backend is enabled. If cmake
    fails to set the `-fopenacc` flag, it can be set manually.
- nvhpc, target tesla (set `$CC`, `$CXX` and `$CUDA_HOME` to appropriate values
  for your system to use nvhpc):
  ```bash
    -DCMAKE_CXX_FLAGS="-acc -ta=tesla -Minfo"
  ```
  - known issues: activating optimizations (other than default `-O`) and debug
    symbols `-g` breaks block-level synchronization and shared memory (nvhpc
    21.9)

## Limitations

* *No separable compilation*. OpenACC requires functions for which device code
  should be generated for a not-inlined call in a target region to be marked with
  pragmas. This cannot be wrapped by macros like `ALPAKA_FN_DEVICE` because they
  appear between template parameter list and function name.
  <https://github.com/alpaka-group/alpaka/pull/1126#discussion_r479761867>

## Test targets

### helloWorld

```bash
make helloWorld
./examples/helloWorld/helloWorld
```
The output should end with something like
```
[z:3, y:7, x:15][linear:511] Hello World
```
Numbers can vary when teams are executed in parallel: 512 teams, with one worker
each are started in a 3d grid. Each worker reports its grid coordinates and linear
index.

|compiler|compile status|target|run status|
|---|---|---|---|
|GCC 10| ok|x86|ok|
|NVHPC 20.7| ok|tesla|ok|

### vectorAdd

```bash
make vectorAdd
./examples/vectorAdd/vectorAdd
```
The output should end with
```
Execution results correct!
```

|compiler|compile status|target|run status|
|---|---|---|---|
|GCC 10(dev)| ok|x86|ok|
|NVHPC 20.7| ok|tesla|ok|

## Building and Running all tests

```bash
make
ctest
```
