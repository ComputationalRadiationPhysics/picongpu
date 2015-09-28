Change Log / Release Log for mallocMC
================================================================

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
