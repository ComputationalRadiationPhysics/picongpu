Change Log / Release Log for mallocMC
================================================================

2.5.0crp
--------
**Date:** 2021-02-18

This release removes the native usage of CUDA by alpaka.
Attention: This release depends on an unreleased [alpaka 0.5.0dev](https://github.com/alpaka-group/alpaka/commit/34870a73ecf702069465aa030fbdf301c4d22c61)
version before the heavy alpaka namespace refactoring.

### Changes to mallocMC 2.4.0crp

**Features**
- Port to alpaka #173

**Bug fixes**
- fix HIP support (warpsize, activemask, compile issues) #182
- fix data race and printf issue #189
- fix data races in `Scatter.hpp` #190
- fix clang cuda compile #192

**Misc:**
- Added alpaka subtree and switched to C++14 #176
- Added 3rd party catch.hpp and made CMake find it #179
- Update documentation after switch to alpaka #194
- Update .clang-format and apply clang-format #197

Thanks to Bernhard Manfred Gruber and Rene Widera for contributing to this release!

2.4.0crp
--------
**Date:** 2020-05-28

This release removes the Boost dependency and switched to C++11.

### Changes to mallocMC 2.3.1crp

**Features**
  - Cleaning, remove Boost dependency & C++11 Migration #169

**Bug fixes**
  - Choose the value for the -arch nvcc flag depending on CUDA version #164 #165

**Misc:**
  - Travis CI: GCC 5.5.0 + CUDA 9.1.85 #170
  - Adding headers to projects and applied clang-tidy #171
  - clang-format #172

Thanks to  Sergei Bastrakov, Bernhard Manfred Gruber and Axel Huebl for contributing to this release!

2.3.1crp
--------
**Date:** 2019-02-14

A critical bug was fixed which can result in an illegal memory access.

### Changes to mallocMC 2.3.0crp

**Bug fixes**
 - fix illegal memory access in `XMallocSIMD` #150

**Misc:**
 - CMake: Honor `<packageName>_ROOT` Env Hints #154


2.3.0crp
--------
**Date:** 2018-06-11

This release adds support for CUDA 9 and clang's -x cuda frontend and fixes several bugs.
Global objects have been refactored to separate objects on host and device.

### Changes to mallocMC 2.2.0crp

**Features**
 - CUDA 9 support #144 #145
 - clang++ -x cuda support #133
 - add `destructiveResize` method #136
 - heap as separate object on host and device, no more globals #116
 - use `BOOST_STATIC_CONSTEXPR` where possible #109

**Bug fixes**
 - fix uninitialized pointers #110 #112
 - fix crash in getAvailableSlots #106 #107
 - Fix `uint32_t` cstdint #104 #105
 - fix missing boost include #142
 - fix includes from C headers #121
 - fix missing local size change in `finalizeHeap()` #135
 - check heap pointer in Scatter creation policy #126

**Misc:**
 - better link usage and install docs #141
 - self consistent allocator #140
 - rename some shadowed variables in C++11 mode #108
 - properly enforce `-Werror` in Travis-CI #128
 - update Travis-CI image #119
 - improved docs #125 #127

Thanks to Carlchristian Eckert, René Widera, Axel Huebl and Alexander Grund for contributing to this release!


2.2.0crp
-------------
**Date:** 2015-09-25

This release fixes some minor bugs that occured after the release of 2.1.0crp, adds some documentation and improves the interoperability with other projects and build systems.
We closed all issues documented in
[Milestone *2.2.0crp: Stabilizing the release*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=5&state=closed)

### Changes to mallocMC 2.1.0crp

**Features**
 - the interface now provides the host function `HeapInfoVector getHeapLocations()` to obtain information about the location and size of existing mallocMC-heaps #86

**Bug fixes**
 - the function `getAvailableSlots` was always required in the policy classes, although the implementations might not provide it #89

**Misc:**
 - the code relied on `__TROW` being defined, which is not available in all compilers #91
 - the CMake dependency increased to CMake >= 2.8.12.2 #92
 - a new FindmallocMC.cmake module file is provided at https://github.com/ComputationalRadiationPhysics/cmake-modules #85
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/2.1.0crp...2.2.0crp


2.1.0crp
-------------
**Date:** 2015-02-11

This release fixes some bugs that occured after the release of 2.0.1crp and reduces the interface to improve interoperability with the default CUDA allocator.
We closed all issues documented in
[Milestone *New Features*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=3&state=closed)

### Changes to mallocMC 2.0.1crp

**Features**
 - the possibility to overwrite the default implementation of new/delete and malloc/free was removed #72. **This changes the interface**, since users are now always forced to call `mallocMC::malloc()` and `mallocMC::free()`. This is intended to improve readability and allows to use the CUDA allocator inside mallocMC.
 - the policy *Scatter* now places the onpagetables data structure at the end of a page. This can greatly improve performance when using large pages and `resetfreedpages=true` #80

**Bug fixes**
 - in the policy *Scatter*, `fullsegments` and `additional_chunks` could grow too large in certain configurations #79

**Misc:**
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/2.0.1crp...2.1.0crp


2.0.1crp
-------------
**Date:** 2015-01-13

This release fixes several bugs that occured after the release of 2.0.0crp.
We closed all issues documented in
[Milestone *Bugfixes*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=4&state=closed)

### Changes to mallocMC 2.0.0crp

**Bug fixes**
 - page table metadata was not correctly initialized with 0 #70
 - freeing pages would not work under certain circumstances #66
 - the bitmask in a page table entry could be wrong due to a racecondition #62
 - not all regions were initialized correctly #60
 - getAvailableSlots could sometimes miss blocks #59
 - the counter for elements in a page could get too high due to a racecondition #61
 - Out of Memory (OOM) Policy sometimes did not recognize allocation failures correctly #67

**Misc:**
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/2.0.0crp...2.0.1crp


2.0.0crp
-------------
**Date:** 2014-06-02

This release introduces mallocMC, which contains the previous algorithm and
much code from ScatterAlloc 1.0.2crp. The project was renamed due to massive
restructurization and because the code uses ScatterAlloc as a reference
algorithm, but can be extended to include other allocators in the future.
We closed all issues documented in
[Milestone *Get Lib ready for PIConGPU*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=2&state=closed)

### Changes to ScatterAlloc 1.0.2crp

**Features**
 - completely split into policies #17
 - configuration through structs instead of macro #17
 - function `getAvailableSlots()` #5
 - selectable data alignment #14
 - function `finalizeHeap()` #11

**Bug fixes:**
 - build warning for cmake #33

**Misc:**
 - verification code and examples #35
 - install routines #4
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/1.0.2crp...2.0.0crp


1.0.2crp
-------------
**Date:** 2014-01-07

This is our first bug fix release.
We closed all issues documented in
[Milestone *Bug fixes*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=1&state=closed)

### Changes to 1.0.1

**Features:**
  - added travis-ci.org support for compile tests #7

**Bug fixes:**
  - broken cmake/compile #1
  - g++ warnings #10
  - only N-1 access blocks used instead of N #2
  - 32bit bug: allocate more than 4GB #12

**Misc:**
  See the full changes at
  https://github.com/ComputationalRadiationPhysics/scatteralloc/compare/1.0.1...1.0.2crp
