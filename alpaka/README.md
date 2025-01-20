**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================


[![Continuous Integration](https://github.com/alpaka-group/alpaka/workflows/Continuous%20Integration/badge.svg)](https://github.com/alpaka-group/alpaka/actions?query=workflow%3A%22Continuous+Integration%22)
[![Documentation Status](https://readthedocs.org/projects/alpaka/badge/?version=latest)](https://alpaka.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://alpaka-group.github.io/alpaka)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20mac-lightgrey.svg)](https://github.com/alpaka-group/alpaka)
[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

![alpaka](docs/logo/alpaka_401x135.png)

The **alpaka** library is a header-only C++17 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

It is platform independent and supports the concurrent and cooperative use of multiple devices such as the hosts CPU (x86, ARM, RISC-V and Power 8+) and  GPU accelerators from different vendors (NVIDIA, AMD and Intel).
A multitude of accelerator back-end variants using NVIDIA CUDA, AMD HIP, SYCL, OpenMP 2.0+, std::thread and also serial execution is provided and can be selected depending on the device.
Only one implementation of the user kernel is required by representing them as function objects with a special interface.
There is no need to write special CUDA, HIP, OpenMP or custom threading code.
Accelerator back-ends can be mixed and synchronized via compute device queue.
The decision which accelerator back-end executes which kernel can be made at runtime.

The abstraction used is very similar to the CUDA grid-blocks-threads domain decomposition strategy.
Algorithms that should be parallelized have to be divided into a multi-dimensional grid consisting of small uniform work items.
These functions are called kernels and are executed in parallel threads.
The threads in the grid are organized in blocks.
All threads in a block are executed in parallel and can interact via fast shared memory and low level synchronization methods.
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

| Accelerator Back-end   | Lib/API                                                 | Devices                | Execution strategy grid-blocks     | Execution strategy block-threads     |
|------------------------|---------------------------------------------------------|------------------------|------------------------------------|--------------------------------------|
| Serial                 | n/a                                                     | Host CPU (single core) | sequential                         | sequential (only 1 thread per block) |
| OpenMP 2.0+ blocks     | OpenMP 2.0+                                             | Host CPU (multi core)  | parallel (preemptive multitasking) | sequential (only 1 thread per block) |
| OpenMP 2.0+ threads    | OpenMP 2.0+                                             | Host CPU (multi core)  | sequential                         | parallel (preemptive multitasking)   |
| std::thread            | std::thread                                             | Host CPU (multi core)  | sequential                         | parallel (preemptive multitasking)   |
| TBB                    | TBB 2.2+                                                | Host CPU (multi core)  | parallel (preemptive multitasking) | sequential (only 1 thread per block) |
| CUDA                   | CUDA 9.0+                                               | NVIDIA GPUs            | parallel (undefined)               | parallel (lock-step within warps)    |
| HIP(clang)             | [HIP 5.1+](https://github.com/ROCm-Developer-Tools/HIP) | AMD GPUs               | parallel (undefined)               | parallel (lock-step within warps)    |


Supported Compilers
-------------------

This library uses C++17 (or newer when available).

| Accelerator Back-end | gcc 9.5 (Linux)                           | gcc 10.4 / 11.1 (Linux)                   | gcc 12.3 (Linux)                      | gcc 13.1 (Linux)                      | clang 9 (Linux)                           | clang 10/11 (Linux)                             | clang 12 (Linux)                          | clang 13 (Linux)                      | clang 14 (Linux)                      | clang 15 (Linux)                      | clang 16 (Linux)                      | clang 17 (Linux)                      | icpx 2024.2 (Linux)     | Xcode 13.2.1 / 14.2 / 14.3.1 (macOS) | Visual Studio 2022 (Windows) |
|----------------------|-------------------------------------------|-------------------------------------------|---------------------------------------|---------------------------------------|-------------------------------------------|-------------------------------------------------|-------------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|-------------------------|--------------------------------------|------------------------------|
| Serial               | :white_check_mark:                        | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                        | :white_check_mark:                              | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:      | :white_check_mark:                   | :white_check_mark:           |
| OpenMP 2.0+ blocks   | :white_check_mark:                        | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                        | :white_check_mark:                              | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark: [^1] | :white_check_mark:                   | :white_check_mark:           |
| OpenMP 2.0+ threads  | :white_check_mark:                        | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                        | :white_check_mark:                              | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark: [^1] | :white_check_mark:                   | :white_check_mark:           |
| std::thread          | :white_check_mark:                        | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                        | :white_check_mark:                              | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:      | :white_check_mark:                   | :white_check_mark:           |
| TBB                  | :white_check_mark:                        | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                        | :white_check_mark:                              | :white_check_mark:                        | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:                    | :white_check_mark:      | :white_check_mark:                   | :white_check_mark:           |
| CUDA (nvcc)          | :white_check_mark: (CUDA 11.2 - 12.5)[^2] | :white_check_mark: (CUDA 11.4 - 12.0)[^2] | :white_check_mark: (CUDA 12.0 - 12.5) | :white_check_mark: (CUDA 12.4 - 12.5) | :white_check_mark: (CUDA 11.6 - 12.0)[^2] | :white_check_mark: (CUDA 11.2, 11.6 - 12.0)[^2] | :white_check_mark: (CUDA 11.6 - 12.0)[^2] | :white_check_mark: (CUDA 11.7 - 12.0) | :white_check_mark: (CUDA 11.8 - 12.0) | :white_check_mark: (CUDA 12.2)        | :white_check_mark: (CUDA 12.3)        | :white_check_mark: (CUDA 12.4 - 15.5) | :x:                     | -                                    | :x:                          |
| CUDA (clang)         | -                                         | -                                         | -                                     | -                                     | :x:                                       | :x:                                             | :x:                                       | :x:                                   | :white_check_mark: (CUDA 11.2 - 11.5) | :white_check_mark: (CUDA 11.2 - 11.5) | :white_check_mark: (CUDA 11.2 - 11.5) | :white_check_mark: (CUDA 11.2 - 11.8) | :x:                     | -                                    | -                            |
| HIP (clang)          | -                                         | -                                         | -                                     | -                                     | :x:                                       | :x:                                             | :x:                                       | :x:                                   | :white_check_mark: (HIP 5.1 - 5.2)    | :white_check_mark: (HIP 5.3 - 5.4)    | :white_check_mark: (HIP 5.5 - 5.6)    | :white_check_mark: (HIP 5.7 - 6.1)    | :x:                     | -                                    | -                            |
| SYCL                 | :x:                                       | :x:                                       | :x:                                   | :x:                                   | :x:                                       | :x:                                             | :x:                                       | :x:                                   | :x:                                   | :x:                                   | :x:                                   | :x:                                   | :white_check_mark: [^4] | -                                    | :x:                          |

Other compilers or combinations marked with :x: in the table above may work but are not tested in CI and are therefore not explicitly supported.

[^1]: Due to an [LLVM bug](https://github.com/llvm/llvm-project/issues/58491) in debug mode only release builds are supported.
[^2]: Due to a [CUDA bug](https://github.com/alpaka-group/alpaka/issues/2035) debug builds are only supported for CUDA versions >= 11.7.
[^3]: Due to an [`icpx` bug](https://github.com/intel/llvm/issues/10711) the OpenMP back-ends cannot be used together with the SYCL back-end.
[^4]: Currently, the unit tests are compiled but not executed.

Dependencies
------------

[Boost](https://boost.org/) 1.74.0+ is the only mandatory external dependency.
The **alpaka** library itself just requires header-only libraries.
However some of the accelerator back-end implementations require different boost libraries to be built.

When an accelerator back-end using *CUDA* is enabled, version *11.2* (with nvcc as CUDA compiler) or version *11.2* (with clang as CUDA compiler) of the *CUDA SDK* is the minimum requirement.
*NOTE*: When using clang as a native *CUDA* compiler, the *CUDA accelerator back-end* can not be enabled together with any *OpenMP accelerator back-end* because this combination is currently unsupported.
*NOTE*: Separable compilation is disabled by default and can be enabled via the CMake flag `CMAKE_CUDA_SEPARABLE_COMPILATION`.

When an accelerator back-end using *OpenMP* is enabled, the compiler and the platform have to support the corresponding minimum *OpenMP* version.

When an accelerator back-end using *TBB* is enabled, the compiler and the platform have to support the corresponding minimum *TBB* version.


Usage
-----

The library is header only so nothing has to be built.
CMake 3.22+ is required to provide the correct defines and include paths.
Just call `alpaka_add_executable` instead of `add_executable` and the difficulties of the CUDA nvcc compiler in handling `.cu` and `.cpp` files are automatically taken care of.
Source files do not need any special file ending.
Examples of how to utilize alpaka within CMake can be found in the `example` folder.

The whole alpaka library can be included with: `#include <alpaka/alpaka.hpp>`
Code that is not intended to be utilized by the user is hidden in the `detail` namespace.

Furthermore, for a CUDA-like experience when adopting alpaka we provide the library [*cupla*](https://github.com/alpaka-group/cupla).
It enables a simple and straightforward way of porting existing CUDA applications to alpaka and thus to a variety of accelerators.

### Single header

The CI creates a single-header version of alpaka on each commit,
which you can find on the [single-header branch](https://github.com/alpaka-group/alpaka/tree/single-header).

This is especially useful, if you would like to play with alpaka on [Compiler explorer](https://godbolt.org/z/hzPnhnna9).
Just include alpaka like
```c++
#include <https://raw.githubusercontent.com/alpaka-group/alpaka/single-header/include/alpaka/alpaka.hpp>
```
and enable the desired backend on the compiler's command line using the corresponding macro, e.g. via `-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`.

Introduction
------------

For a quick introduction, feel free to playback the recording of our presentation at
[GTC 2016](https://www.nvidia.com/gtc/):

 - E. Zenker, R. Widera, G. Juckeland et al.,
   *Porting the Plasma Simulation PIConGPU to Heterogeneous Architectures with Alpaka*,
   [video link (39 min)](http://on-demand.gputechconf.com/gtc/2016/video/S6298.html),
   [slides (PDF)](https://on-demand.gputechconf.com/gtc/2016/presentation/s6298-erik-zenker-porting-the-plasma.pdf),
   [DOI:10.5281/zenodo.6336086](https://doi.org/10.5281/zenodo.6336086)


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

Rules for contributions can be found in [CONTRIBUTING.md](CONTRIBUTING.md).
Any pull request will be reviewed by a [maintainer](https://github.com/orgs/alpaka-group/teams/alpaka-maintainers).

Thanks to all [active and former contributors](.zenodo.json).
