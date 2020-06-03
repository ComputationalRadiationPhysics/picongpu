Changelog
=========

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

