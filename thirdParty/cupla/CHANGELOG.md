Changelog
=========

0.4.0 "K'Ehleyr"
----------------
**Date:** 2022-05-19

### Compatibility Changes:

- switch to alpaka 0.9.X #225 #226, [see all breaking changes](https://github.com/alpaka-group/alpaka/blob/0.9.0/CHANGELOG.md#breaking-changes)
- modernize CMake #203

### Bug Fixes:
- allow including alpaka via add_subdirectory before cupla in an external project #216
- fix using cupla with add_subdirectory for the HIP backend #219
- fix warning: expression result unused #222

### New Features:
- documenting interoperability with alpaka API

### Misc
- clang format integration #214

0.3.0 "L'Rell"
--------------
**Date:** 2021-08-03

### Compatibility Changes:

- switch to alpaka 0.6.X (heavy alpaka namespace refactoring) #184 #186 #192 #194 #199 #201
- usage of cupla without cuda renaming macros #161 #179
- move `cupla/math` to `cupla/device/math` #166
- refactor math functions #162 #168

### Bug Fixes:
- fix KernelWithElementLevel documentation #159
- fix atomic function return type #163
- fix ambiguous namespace `device` #167
- fix ; warnings #172
- fix multi GPU streams handling and default stream #183

### New Features:
- math functions: erf and pow #169
- add support for Alpaka's OpenMP 5 and OpenACC backends #196

### Misc
- CI tests for AMD GPUs #187
- check for supported alpaka version #197
- CI: remove travis tests #202

0.2.0 "Khan"
------------
**Date:** 2019-02-18

- update alpaka to 0.4.0 #128 #143
- support `cudaGetErrorName` #113
- support `cudaEventBlockingSync` #98
- support for alpaka OMP4 backend #137
- support for alpaka TBB backend #105
- support for alpaka HIP backend #97
- adds `atomicAnd`, `atomicXor`, `atomicOr` #117
- adds float3 and int3 with make-functions. #154
- configuration header/cupla support without CMake #112
- move all accelerator-specific code to an inline namespace #106
- pin memory allocated with `cuplaMallocHost` #144
- switch internally shipped version of alpaka from a git submodule to subtree #103
- refactor cupla kernel execution #136
- compile cupla interfaces into a static library #140
- add Black-Scholes example #153

0.1.1 "Colonel Worf"
--------------------
**Date:** 2019-01-23

- update Alpaka submodule to 0.3.5 #81
- add `cuplaGetErrorString` #84
- add `cuplaStreamQuerry` #85
- add `cuplaPeekAtLastError` #86
- fix TBB backend activation #90

0.1.0 "Worf"
------------
**Date:** 2018-06-10

Worf: Peace between the Worlds

Following the shining example of Worf, the 0.1.0 release of cupla makes peace
between the worlds of GPU and CPU computing. The powerful Lords of CUDA
computing can now speak seamlessly to the inferior worlds of CPUs.

**Supported Alpaka Versions:** 0.3.X

