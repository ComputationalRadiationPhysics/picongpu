**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================


[![Continuous Integration](https://github.com/alpaka-group/alpaka/workflows/Continuous%20Integration/badge.svg)](https://github.com/alpaka-group/alpaka/actions?query=workflow%3A%22Continuous+Integration%22)
[![Documentation Status](https://readthedocs.org/projects/alpaka/badge/?version=latest)](https://alpaka.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://alpaka-group.github.io/alpaka)
[![Language](https://img.shields.io/badge/language-C%2B%2B14-orange.svg)](https://isocpp.org/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20mac-lightgrey.svg)](https://github.com/alpaka-group/alpaka)
[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

![alpaka](docs/logo/alpaka_401x135.png)

The **alpaka** library is a header-only C++14 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

It is platform independent and supports the concurrent and cooperative use of multiple devices such as the hosts CPU as well as attached accelerators as for instance CUDA GPUs and Xeon Phis (currently native execution only).
A multitude of accelerator back-end variants using CUDA, OpenMP (2.0/4.0), Boost.Fiber, std::thread and also serial execution is provided and can be selected depending on the device.
Only one implementation of the user kernel is required by representing them as function objects with a special interface.
There is no need to write special CUDA, OpenMP or custom threading code.
Accelerator back-ends can be mixed within a device queue.
The decision which accelerator back-end executes which kernel can be made at runtime.

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

**alpaka** is licensed under **MPL-2.0**.


Documentation
-------------

The alpaka documentation can be found in the [online manual](https://alpaka.readthedocs.io).
The documentation files in [`.rst` (reStructuredText)](https://www.sphinx-doc.org/en/stable/rest.html) format are located in the `docs` subfolder of this repository.
The [source code documentation](https://alpaka-group.github.io/alpaka/) is generated with [doxygen](http://www.doxygen.org).


Accelerator Back-ends
---------------------

|Accelerator Back-end|Lib/API|Devices|Execution strategy grid-blocks|Execution strategy block-threads|
|---|---|---|---|---|
|Serial|n/a|Host CPU (single core)|sequential|sequential (only 1 thread per block)|
|OpenMP 2.0+ blocks|OpenMP 2.0+|Host CPU (multi core)|parallel (preemptive multitasking)|sequential (only 1 thread per block)|
|OpenMP 2.0+ threads|OpenMP 2.0+|Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
|OpenMP 5.0+ |OpenMP 5.0+|Host CPU (multi core)|parallel (undefined)|parallel (preemptive multitasking)|
| ||GPU|parallel (undefined)|parallel (lock-step within warps)|
|OpenACC (experimental)|OpenACC 2.0+|Host CPU (multi core)|parallel (undefined)|parallel (preemptive multitasking)|
|||GPU|parallel (undefined)|parallel (lock-step within warps)|
| std::thread | std::thread |Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
| Boost.Fiber | boost::fibers::fiber |Host CPU (single core)|sequential|parallel (cooperative multitasking)|
|TBB|TBB 2.2+|Host CPU (multi core)|parallel (preemptive multitasking)|sequential (only 1 thread per block)|
|CUDA|CUDA 9.0+|NVIDIA GPUs|parallel (undefined)|parallel (lock-step within warps)|
|HIP(clang)|[HIP 3.5+](https://github.com/ROCm-Developer-Tools/HIP)|AMD GPUs |parallel (undefined)|parallel (lock-step within warps)|


Supported Compilers
-------------------

This library uses C++14 (or newer when available).

|Accelerator Back-end|gcc 5.5 <br/> (Linux)|gcc 6.4/7.3 <br/> (Linux)|gcc 8.1 <br/> (Linux)|gcc 9.1 <br/> (Linux)|gcc 10.1 <br/> (Linux)|clang 4 <br/> (Linux)|clang 5 <br/> (Linux)|clang 6 <br/> (Linux)|clang 7 <br/> (Linux)|clang 8 <br/> (Linux)|clang 9 <br/> (Linux)|clang 10 <br/> (Linux)|Apple LLVM 11.2.1-12.2.0 <br/> (macOS)|MSVC 2017 <br/> (Windows)|MSVC 2019 <br/> (Windows)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Serial|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0+ blocks|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0+ threads|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
|OpenMP 4.0+ (CPU)|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:x:|:x:|
| std::thread |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
| Boost.Fiber |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
|TBB|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:x:|
|CUDA (nvcc)|:white_check_mark: <br/> (CUDA 9.0-11.1)|:white_check_mark: <br/> (CUDA 9.2-11.1) |:white_check_mark: <br/> (CUDA 10.1-11.1) |:white_check_mark: <br/> (CUDA 11.0-11.1)|:white_check_mark: <br/> (CUDA 11.1)|:white_check_mark: <br/> (CUDA 9.1-11.1)|:white_check_mark: <br/> (CUDA 10.1-11.1)|:white_check_mark: <br/> (CUDA 10.1-11.1)|:white_check_mark: <br/> (CUDA 10.1-11.1)|:white_check_mark: <br/> (CUDA 10.1-11.1)|:white_check_mark: <br/> (CUDA 11.0-11.1)|:white_check_mark: <br/> (CUDA 11.1)|:x:|:white_check_mark: <br/> (CUDA 10.0-11.1)|:white_check_mark: <br/> (CUDA 10.1-10.2 + 11.1)|
|CUDA (clang) | - | - | - | - | - | - | - | :white_check_mark: <br/> (CUDA 9.0) | :white_check_mark: <br/> (CUDA 9.0-9.2) | :white_check_mark: <br/> (CUDA 9.0-10.0) | :white_check_mark: <br/> (CUDA 9.2-10.1) | :white_check_mark: <br/> (CUDA 9.2-10.1) | - | - | - |
|[HIP](https://alpaka.readthedocs.io/en/latest/install/HIP.html) (clang)|:white_check_mark: |:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|:x:|


Other compilers or combinations marked with :x: in the table above may work but are not tested in CI and are therefore not explicitly supported.

Dependencies
------------

[Boost](https://boost.org/) 1.65.1+ is the only mandatory external dependency.
The **alpaka** library itself just requires header-only libraries.
However some of the accelerator back-end implementations require different boost libraries to be built.

When an accelerator back-end using *Boost.Fiber* is enabled, `boost-fiber` and all of its dependencies are required to be built in C++14 mode `./b2 cxxflags="-std=c++14"`.
When *Boost.Fiber* is enabled and alpaka is built in C++17 mode with clang and libstc++, Boost >= 1.67.0 is required.

When an accelerator back-end using *CUDA* is enabled, version *9.0* of the *CUDA SDK* is the minimum requirement.
*NOTE*: When using nvcc as *CUDA* compiler, the *CUDA accelerator back-end* can not be enabled together with the *Boost.Fiber accelerator back-end* due to bugs in the nvcc compiler.
*NOTE*: When using clang as a native *CUDA* compiler, the *CUDA accelerator back-end* can not be enabled together with the *Boost.Fiber accelerator back-end* or any *OpenMP accelerator back-end* because this combination is currently unsupported.
*NOTE*: Separable compilation is only supported when using nvcc, not with clang as native *CUDA* compiler. It is disabled by default and can be enabled via the CMake flag `ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION`.

When an accelerator back-end using *OpenMP* is enabled, the compiler and the platform have to support the corresponding minimum *OpenMP* version.

When an accelerator back-end using *TBB* is enabled, the compiler and the platform have to support the corresponding minimum *TBB* version.


Usage
-----

The library is header only so nothing has to be built.
CMake 3.15+ is required to provide the correct defines and include paths.
Just call `ALPAKA_ADD_EXECUTABLE` instead of `CUDA_ADD_EXECUTABLE` or `ADD_EXECUTABLE` and the difficulties of the CUDA nvcc compiler in handling `.cu` and `.cpp` files are automatically taken care of.
Source files do not need any special file ending.
Examples of how to utilize alpaka within CMake can be found in the `example` folder.

The whole alpaka library can be included with: `#include <alpaka/alpaka.hpp>`
Code that is not intended to be utilized by the user is hidden in the `detail` namespace.

Furthermore, for a CUDA-like experience when adopting alpaka we provide the library [*cupla*](https://github.com/alpaka-group/cupla).
It enables a simple and straightforward way of porting existing CUDA applications to alpaka and thus to a variety of accelerators.

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

Contributing
------------

Rules for contributions can be found in [CONTRIBUTING.md](CONTRIBUTING.md)

Authors
-------

### Maintainers* and Core Developers

- Benjamin Worpitz* (original author)
- Dr. Sergei Bastrakov*
- Simeon Ehrig
- Bernhard Manfred Gruber
- Dr. Axel Huebl*
- Dr. Jeffrey Kelling
- Jakob Krude
- Jan Stephan*
- Rene Widera*

### Former Members, Contributions and Thanks

- Dr. Michael Bussmann
- Mat Colgrove
- Valentin Gehrke
- Maximilian Knespel
- Alexander Matthes
- Hauke Mewes
- Phil Nash
- Mutsuo Saito
- Jonas Schenke
- Daniel Vollmer
- Matthias Werner
- Bert Wesarg
- Malte Zacharias
- Erik Zenker
