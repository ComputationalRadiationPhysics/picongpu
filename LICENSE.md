PIConGPU - Licenses
===============================================================================

**Copyright 2009-2018** (in alphabetical order)

Sergei Bastrakov, Florian Berninger, Heiko Burau, Michael Bussmann,
Alexander Debus, Robert Dietrich, Carlchristian Eckert, Wen Fu, Marco Garten,
Ilja Goethel, Alexander Grund, Sebastian Hahn, Anton Helm, Wolfgang Hoehnig,
Axel Huebl, Jeffrey Kelling, Maximilian Knespel, Remi Lehe, Alexander Matthes,
Richard Pausch, Sophie Rudat, Felix Schmitt, Conrad Schumann,
Benjamin Schneider, Joseph Schuchart, Sebastian Starke, Klaus Steiniger,
Rene Widera, Benjamin Worpitz

See [active team](README.md#active-team).

PIConGPU is a program collection containing the main simulation, independent
scripts and auxiliary libraries. If not stated otherwise explicitly, the
following licenses apply:


### PIConGPU

The **main simulation** is licensed under the **GPLv3+**. See
[COPYING](COPYING). If not stated otherwise explicitly, that affects:
 - `buildsystem`
 - `etc/picongpu`
 - `include/picongpu`
 - `lib/python/picongpu`
 - `share/picongpu`
 - `src/tools` (without `splash2txt`)
 - `src/mpiInfo`
 - `test`


### PMacc & splash2txt
 
All **libraries** are licensed under the **LGPLv3+**. See
[COPYING.LESSER](COPYING.LESSER).
If not stated otherwise explicitly, that affects:
 - `include/pmacc`
 - `share/pmacc`
 - `src/tools/splash2txt`


### Documentation

Documentation is licensed under CC-BY 4.0.
See https://creativecommons.org/licenses/by/4.0/ for the license.

If not stated otherwise explicitly, that affects files in:

- `docs`


### Third party software and other licenses

We include a list of (GPL-) compatible third party software for the sake
of an easier install of `PIConGPU`. Contributions to these parts of the
repository should *not* be made in the `thirdParty/` directory but in
*their according repositories* (that we import).

 - `thirdParty/mallocMC`:
   mallocMC is a fast memory allocator for many core accelerators and was
   originally forked from the `ScatterAlloc` project.
   It is licensed under the *MIT License*.
   Please visit
     https://github.com/ComputationalRadiationPhysics/mallocMC
   for further details and contributions.

 - `thirdParty/cmake-modules`:
   we published a set of useful CMake modules that are not in the
   CMake mainline under the *ISC license* at
     https://github.com/ComputationalRadiationPhysics/cmake-modules
   for contributions or inclusion in PIConGPU and other projects.

 - `thirdParty/cuda_memtest`:
   CUDA MemTest is an *independent program* developed by the University
   Illinois, published under the *Illinois Open Source License*.
   Please refer to the file `thirdParty/cuda_memtest/README` for license information.
   We redistribute this modified version of CUDA MemTest under the same license
   [thirdParty/cuda_memtest/README](thirdParty/cuda_memtest/README).
   The original release was published at
     http://sourceforge.net/projects/cudagpumemtest
   and our modified version is hosted at
     https://github.com/ComputationalRadiationPhysics/cuda_memtest
   for further reference.

- `thirdParty/cupla`:
   cupla is a simple user interface for alpaka. It provides a software layer
   that follows a similar concept as the Nvidia CUDA API, allowing to write
   kernels more efficiently.
   Please visit
     https://github.com/ComputationalRadiationPhysics/cupla
   for further details and contributions.

- `thirdParty/nlohmann_json`:
   nlohmann_json is a modern C++ library for working with JSON data, developed
   by Niels Lohmann, published under the MIT License.
   Please refer to the file `thirdParty/nlohmann_json/LICENSE.MIT` for license
   information.
   Please visit https://github.com/nlohmann/json for further details
   and contributions.
