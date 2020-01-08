**cupla** - C++ User interface for the Platform independent Library Alpaka
==========================================================================

[![Build Status dev](https://img.shields.io/travis/ComputationalRadiationPhysics/cupla/dev.svg?label=dev)](https://travis-ci.org/ComputationalRadiationPhysics/cupla/branches)

**cupla** [[qχɑpˈlɑʔ]](https://en.wiktionary.org/wiki/Qapla%27) is a simple user
interface for the platform independent parallel kernel
acceleration library
[**alpaka**](https://github.com/ComputationalRadiationPhysics/alpaka).
It follows a similar concept as the
[NVIDIA® CUDA® API](https://developer.nvidia.com/cuda-zone) by
providing a software layer to manage accelerator devices.
**alpaka** is used as backend for **cupla**.

Please keep in mind that a first, ["find & replace"](doc/PortingGuide.md) port
from **CUDA to cupla(x86)** will result in rather bad performance. In order to
reach decent performance on x86 systems you just need to add the **alpaka**
[element level](doc/TuningGuide.md) to your kernels.

(*Read as:* add some *tiling* to your CUDA kernels by letting the same thread
compute a fixed number of elements (N=4..16) instead of just computing one
*element* per thread. Also, make the number of elements in your tiling a
*compile-time constant* and your CUDA code (N=1) will just stay with the
very same performance while adding single-source performance portability for,
e.g., x86 targets).


Software License
----------------

**cupla** is licensed under **LGPLv3** or later.

For more information see [LICENSE.md](LICENSE.md).


Dependencies
------------

- **cmake 3.11.0**
- **[alpaka](https://github.com/ComputationalRadiationPhysics/alpaka/)**
  - alpaka is loaded as `git subtree` within **cupla**, see [INSTALL.md](INSTALL.md)

Usage
-----

- See our notes in [INSTALL.md](INSTALL.md).
- Checkout the [guide](doc/PortingGuide.md) how to port your project.
- Checkout the [tuning guide](doc/TuningGuide.md) for a step further to performance
  portable code.

[cupla can be used as a header-only library and without the CMake build system](doc/ConfigurationHeader.md)

Authors
-------

### Maintainers and core developers

- Rene Widera
- Matthias Werner

### Former Members, Contributions and Thanks

- Axel Huebl
- Dr. Michael Bussmann


Trademarks Disclaimer
---------------------

All product names and trademarks are the property of their respective owners.
CUDA® is a trademark of the NVIDIA Corporation.
