**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

[![Travis CI Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=develop)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/xjeyugcg1cb0662s/branch/develop?svg=true)](https://ci.appveyor.com/project/BenjaminW3/alpaka-vuiya/branch/develop)
[![Language](https://img.shields.io/badge/language-C%2B%2B11-orange.svg)](https://isocpp.org/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey.svg)](https://github.com/ComputationalRadiationPhysics/alpaka)
[![License](https://img.shields.io/badge/license-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.de.html)

The **alpaka** library is a header-only C++11 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

It is platform independent and supports the concurrent and cooperative use of multiple devices such as the hosts CPU as well as attached accelerators as for instance CUDA GPUs and Xeon Phis (currently native execution only).
A multitude of accelerator back-end variants using CUDA, OpenMP (2.0/4.0), Boost.Fiber, std::thread and also serial execution is provided and can be selected depending on the device.
Only one implementation of the user kernel is required by representing them as function objects with a special interface.
There is no need to write special CUDA, OpenMP or custom threading code.
Accelerator back-ends can be mixed within a device stream.
The decision which accelerator back-end executes which kernel can be made at runtime.

The **alpaka** API is currently unstable (beta state).

The abstraction used is very similar to the CUDA grid-blocks-threads division strategy.
Algorithms that should be parallelized have to be divided into a multi-dimensional grid consisting of small uniform work items.
These functions are called kernels and are executed in parallel threads.
The threads in the grid are organized in blocks.
All threads in a block are executed in parallel and can interact via fast shared memory.
Blocks are executed independently and can not interact in any way.
The block execution order is unspecified and depends on the accelerator in use.
By using this abstraction the execution can be optimally adapted to the available hardware.


Software License
----------------

**alpaka** is licensed under **LGPLv3** or later.


Documentation
-------------

The [general documentation](doc/markdown/Index.md) is located within the `doc/markdown` subfolder of the repository.
The [source code documentation](http://computationalradiationphysics.github.io/alpaka/) is generated with [doxygen](http://www.doxygen.org).


Accelerator Back-ends
---------------------

|Accelerator Back-end|Lib/API|Devices|Execution strategy grid-blocks|Execution strategy block-threads|
|---|---|---|---|---|
|Serial|n/a|Host CPU (single core)|sequential|sequential (only 1 thread per block)|
|OpenMP 2.0+ blocks|OpenMP 2.0+|Host CPU (multi core)|parallel (preemptive multitasking)|sequential (only 1 thread per block)|
|OpenMP 2.0+ threads|OpenMP 2.0+|Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
|OpenMP 4.0+ (CPU)|OpenMP 4.0+|Host CPU (multi core)|parallel (undefined)|parallel (preemptive multitasking)|
| std::thread | std::thread |Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
| Boost.Fiber | boost::fibers::fiber |Host CPU (single core)|sequential|parallel (cooperative multitasking)|
|TBB|TBB 2.2+|Host CPU (multi core)|parallel (preemptive multitasking)|sequential (only 1 thread per block)|
|CUDA|CUDA 7.0-9.2|NVIDIA GPUs|parallel (undefined)|parallel (lock-step within warps)|


Supported Compilers
-------------------

This library uses C++11 (or newer when available).

|Accelerator Back-end|gcc 4.9.2|gcc 5.4|gcc 6.3/7.2|clang 3.5/3.6|clang 3.7/3.8|clang 3.9|clang 4|clang 5|MSVC 2017.5|
|---|---|---|---|---|---|---|---|
|Serial|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0+ blocks|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0+ threads|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 4.0+ (CPU)|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
| std::thread |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
| Boost.Fiber |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|TBB|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|CUDA (nvcc)|:white_check_mark: <br/> (CUDA 7.0-9.2)|:white_check_mark: <br/> (CUDA 9.0-9.2)|:white_check_mark: <br/> (CUDA 9.2)|:white_check_mark: <br/> (CUDA 8.0)|:x:|:white_check_mark: <br/> (CUDA 9.1-9.2)|:white_check_mark: <br/> (CUDA 9.1-9.2)|:x:|:x:|
|CUDA (clang) | - | - | - | - | - | - | :white_check_mark: <br/> (CUDA 8.0)| :white_check_mark: <br/> (CUDA 8.0) | - |


Dependencies
------------

[Boost](http://boost.org/) 1.62+ is the only mandatory external dependency (for CUDA 9+ Boost >=1.65.1 is required).
The **alpaka** library itself just requires header-only libraries.
However some of the accelerator back-end implementations require different boost libraries to be built.

When an accelerator back-end using *Boost.Fiber* is enabled, `boost-fiber` and all of its dependencies are required to be build in C++11 mode `./b2 cxxflags="-std=c++11"`.

When an accelerator back-end using *CUDA* is enabled, version *7.0* of the *CUDA SDK* is the minimum requirement.
*NOTE*: When using nvcc as *CUDA* compiler, the *CUDA accelerator back-end* can not be enabled together with the *Boost.Fiber accelerator back-end* due to bugs in the nvcc compiler.
*NOTE*: When using clang as a native *CUDA* compiler, the *CUDA accelerator back-end* can not be enabled together with any *OpenMP accelerator back-end* because this combination is currently unsupported.

When an accelerator back-end using *OpenMP* is enabled, the compiler and the platform have to support the corresponding minimum *OpenMP* version.

When an accelerator back-end using *TBB* is enabled, the compiler and the platform have to support the corresponding minimum *TBB* version.


Usage
-----

The library is header only so nothing has to be build.
CMake 3.7.0+ is required to provide the correct defines and include paths.
Just call `ALPAKA_ADD_EXECUTABLE` instead of `CUDA_ADD_EXECUTABLE` or `ADD_EXECUTABLE` and the difficulties of the CUDA nvcc compiler in handling `.cu` and `.cpp` files are automatically taken care of.
Source files do not need any special file ending.
Examples of how to utilize alpaka within CMake can be found in the `example` folder.

The whole alpaka library can be included with: `#include <alpaka/alpaka.hpp>`
Code that is not intended to be utilized by the user is hidden in the `detail` namespace.


Introduction
------------

For a quick introduction, feel free to playback the recording of our presentation at
[GTC 2016](http://mygtc.gputechconf.com/quicklink/858sI36):

 - E. Zenker, R. Widera, G. Juckeland et al.,
   *Porting the Plasma Simulation PIConGPU to Heterogeneous Architectures with Alpaka*,
   [video link (39 min)](http://on-demand.gputechconf.com/gtc/2016/video/S6298.html)


Citing alpaka
-------------

Currently all authors of **alpaka** are scientists or connected with
research. For us to justify the importance and impact of our work, please
consider citing us accordingly in your derived work and publications:

```latex
% Peer-Reviewed Publication %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Peer reviewed and accepted publication in
%   "2nd International Workshop on Performance Portable
%    Programming Models for Accelerators (P^3MA)"
% colocated with the
%   "2017 ISC High Performance Conference"
%   in Frankfurt, Germany
@inproceedings{MathesP3MA2017,
  author    = {{Matthes}, A. and {Widera}, R. and {Zenker}, E. and {Worpitz}, B. and 
               {Huebl}, A. and {Bussmann}, M.},
  title     = {Tuning and optimization for a variety of many-core architectures without changing a single line of implementation code
               using the Alpaka library},
  archivePrefix = "arXiv",
  eprint    = {1706.10086},
  keywords  = {Computer Science - Distributed, Parallel, and Cluster Computing},
  day       = {30},
  month     = {Jun},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.10086},
}

% Peer-Reviewed Publication %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Peer reviewed and accepted publication in
%   "The Sixth International Workshop on
%    Accelerators and Hybrid Exascale Systems (AsHES)"
% at the
%   "30th IEEE International Parallel and Distributed
%    Processing Symposium" in Chicago, IL, USA
@inproceedings{ZenkerAsHES2016,
  author    = {Erik Zenker and Benjamin Worpitz and Ren{\'{e}} Widera
               and Axel Huebl and Guido Juckeland and
               Andreas Kn{\"{u}}pfer and Wolfgang E. Nagel and Michael Bussmann},
  title     = {Alpaka - An Abstraction Library for Parallel Kernel Acceleration},
  archivePrefix = "arXiv",
  eprint    = {1602.08477},
  keywords  = {Computer science;CUDA;Mathematical Software;nVidia;OpenMP;Package;
               performance portability;Portability;Tesla K20;Tesla K80},
  day       = {23},
  month     = {May},
  year      = {2016},
  publisher = {IEEE Computer Society},
  url       = {http://arxiv.org/abs/1602.08477},
}


% Original Work: Benjamin Worpitz' Master Thesis %%%%%%%%%%
%
@MasterThesis{Worpitz2015,
  author = {Benjamin Worpitz},
  title  = {Investigating performance portability of a highly scalable
            particle-in-cell simulation code on various multi-core
            architectures},
  school = {{Technische Universit{\"{a}}t Dresden}},
  month  = {Sep},
  year   = {2015},
  type   = {Master Thesis},
  doi    = {10.5281/zenodo.49768},
  url    = {http://dx.doi.org/10.5281/zenodo.49768}
}
```


Authors
-------

### Maintainers and Core Developers

- Benjamin Worpitz (original author)
- Rene Widera

### Former Members, Contributions and Thanks

- Dr. Michael Bussmann
- Axel Huebl
- Erik Zenker
