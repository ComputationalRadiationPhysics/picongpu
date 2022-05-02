# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.9.0] - 2022-04-21
### Compatibility Changes:
- Platform support added:
  - oneTBB #1456
  - clang 13 #1476
  - CUDA 11.5 #1486
  - Visual Studio 2022 #1583
  - CUDA 11.6 #1616
  - ROCm 5.0 #1631
  - Xcode 12.4 / 13.2.1 #1638
- Platform support removed:
  - CUDA 11.0 / 11.1 + MSVC #1331
  - clang 5 + CUDA #1466
  - Ubuntu 18.04 #1471
  - TBB versions before oneTBB #1456
  - clang 6 / 7 + CUDA #1506
  - Boost < 1.74 #1521
  - CUDA 11.3 - 11.5 + clang #1627
  - Xcode 11.3.1 #1638

### Bug Fixes:
- alpaka TBB kernels are now protected when called from within existing parallel TBB code #1450
- The cheat sheet now reflects the 0.8 changes to alpaka's RNG features #1469
- alpaka `Queue`s will now wait for active asynchronous operations before destructing #1514
- The test cases no longer fail on non-x86 hardware because of `-Werror` #1516
- Several small fixes for the OpenACC and OpenMP 5 back-ends #1564
- Avoid locking in CPU atomic operations #1566
- alpaka's `NormReal` RNG distribution is now copyable (like the other distributions) #1591
- The class layout of `BufCpu` no longer depends on whether the CUDA and HIP back-ends are enabled. #1612
- Fixed several smaller bugs in `alpaka::Vec` #1620
- Destructors no longer throw an exception #1632
- Implemented work-around for Intel compiler bug with OpenMP back-ends #1677

### New Features:
- alpaka now has native complex number support #1336
- alpaka now requires C++17 (or newer). This release therefore includes many refactoring PRs that migrate the code base to C++17:
  - Set CMake requirements, remove versions checks, fix warnings, etc. #1466
  - Removed pre-C++17 workarounds #1483
  - Replaced `alpaka::meta::apply` with `std::apply` #1493
  - Replaced a lot of macros and template metaprogramming sections with `if constexpr` blocks #1495
  - Replaced `alpaka::meta::Void` with `std::void_t` #1499
  - Replaced some of alpaka's metafunctions with their standard counterparts #1501
  - Make use of C++17 mandatory copy elision #1502
  - Simplified CPU kernel launches #1511
  - Make use of generic `std` container interfaces #1554
  - Replaced `std::enable_if` with `if constexpr` where possible #1556
  - Replaced `alpaka::ignore_unused` with C++17 `[[maybe_unused]]` and `std::ignore` #1563
  - Make use of nested namespaces #1587 #1592
  - Make use of variable template versions of `std` traits #1594
- alpaka `Event`s can now be queried for their device type #1479
- Some alpaka buffers can now be allocated asynchronously within a device queue (queue-ordered memory buffers) #1481
  - This capability can be queried with the `hasAsyncBufSupport` trait #1578
- alpaka buffers can now be zero-dimensional (scalar) #1536
- Apply `alpaka::memset` and `alpaka::memcpy` to the whole buffer if no extent is supplied by the user #1547
- Added an accessor-like interface to buffers and views #1570
- Host code utilizing the CUDA and HIP back-ends can now be compiled with a non-CUDA/HIP compiler if there is no device code in the translation unit #1567
- Added `alpaka::getNativeHandle()` to obtain the back-end specific handles from alpaka `Device`s #1579
  - `alpaka::getNativeHandle()` can also be called on `Queue`s and `Event`s #1623
- Added an experimental SYCL back-end. All SYCL back-end functionality currently lives in the `alpaka::experimental` namespace. See the `README_SYCL.md` for more information about the usage and the restrictions of this back-end. #1598
- alpaka's memory fences can now also be applied to the grid level #1641
- `alpaka::getWarpSize()` was renamed to `alpaka::getWarpSizes()` and will now return a `std::vector` of supported warp sizes #1644
- Added previously missing atomic functions for some datatypes #1658
- `ALPAKA_ASSERT` is now variadic #1661
- Documentation updates:
  - Improved installation and usage documentation #1571
  - Added documentation on how to write unit tests #1609
  - The HIP portion of the compiler support matrix has been simplified #1637
  - The OpenMP 5 documentation has been extended #1672

### Misc:
- The CUDA and HIP back-ends no longer explicitly set the device where this is unnecessary #1515
- `clang-tidy`'s modernization suggestions have been applied to the code base #1584
- alpaka's math headers have been squashed. For each back-end there is now only one header instead of one for each math function. #1585
- Updated the Boost predefinition header to reflect the upgrade to Boost 1.74 #1586
- Removed the `alpaka::extent` namespace (the contents now live in the main `alpaka` namespace) #1593
- Refactored implementations of `BufCpu` and `BufUniformCudaHipRt` #1608
- Removed unnecessary specializations of `GetPitchBytes` trait #1614
- All alpaka-specific CMake variables follow the `${PROJECT_NAME}_VARIABLE_FOO_BAR` pattern. This means that all alpaka-specific CMake variables look like this: `alpaka_VARIABLE_FOO_BAR`. #1653
- alpaka now enforces that kernel arguments are trivially copyable #1635
- Renamed namespace `traits` to `trait` #1651
- alpaka now enforces that kernel functions are trivially copyable #1654
- Replaced the internal `hipLaunchKernelGGL()` call with a `kernel<<<...>>>()` call #1663
- `BOOST_LANG_HIP` will now report a (somewhat) correct version number (for internal consumption) #1664
- Refactored `Queue` implementation for CUDA and HIP to reduce code duplication #1667
- `core/CudaHipMath.hpp` was merged back into `math/MathUniformCudaHipBuiltIn.hpp` #1668
- The OpenMP 5 memory fence no longer explicitly sets the `acq_rel` memory order clause since it is the default #1673
- Improved handling of `std::shared_ptr` inside the CUDA/HIP queues #1674
- Internally replaced the deprecated `cudaStreamAddCallback` with `cudaLaunchHostFunc` #1675
- Added CUDA- and HIP-specific aliases for `Event`s, `Platform`s and `Buffer`s #1678

### Breaking Changes
- C++14 is no longer supported (see above)
- alpaka now uses Boost.Atomic by default if the latter can be found by CMake. This can be turned off by passing `-DALPAKA_ACC_CPU_DISABLE_ATOMIC_REF=ON` during the CMake configuration phase. #1566
  - When compiling in C++20 mode, alpaka will use `std::atomic_ref<T>` instead #1671
- Removed the `alpaka::extent` namespace (the contents now live in the main `alpaka` namespace) #1593
- Kernel arguments are required to be trivially copyable. This was always a requirement but is now enforced by alpaka #1635
- All alpaka-specific CMake variables follow the `${PROJECT_NAME}_VARIABLE_FOO_BAR` pattern. This means that all alpaka-specific CMake variables look like this: `alpaka_VARIABLE_FOO_BAR`. #1653
- `alpaka::getWarpSize()` was renamed to `alpaka::getWarpSizes()` and now returns a vector of supported warp sizes #1644
- `alpaka::clock()` is now deprecated and will be removed in the next release. The compiler will warn about its usage. #1645
- Renamed namespace `traits` to `trait` #1651
- Removed support for `std::function` kernel functions #1654
- Kernel functions are required to be trivially copyable. With the exception of `std::function` (see bullet point directly above) this was always a requirement but is now enforced by alpaka. #1654

### Test Cases / CI:
- Added clang 11, 12 + CUDA 11 tests to CI #1466
- Removed Ubuntu 18.04 from CI #1471
- Upgraded all Linux CI runners to Ubuntu 20.04 #1484
- Test whether alpaka can be installed through `cmake --install` #1488
- GitHub CI runners now use all available cores #1508
- Migrated all CUDA runners from GitHub to GitLab #1520
- Refactored `matMul` test #1526
- macOS CI runners now install and test OpenMP 2.x #1533
- Always enable the serial back-end when building the test cases and/or examples #1534
- GitLab CI runners are executed in stages to reduce CI pressure #1537
- Fixed the pitch calculation in `randomCells2D` example #1549
- Updated test infrastructure to Catch2 v2.13.8 #1557
- GitLab CI runners will display all required information to locally reproduce the test environment #1589
- Refactored `alpaka::test` #1596
- `matMul` test will now measure the performance of `alpaka::memcpy` #1599
- The move constructors and assignment operators of buffers are now unit-tested #1611
- Unit tests will now be run with zero dimensionality, too #1619
- Added more tests for `alpaka::Vec` #1633
- Added test for `alpaka::Vec` being trivially copyable #1639
- CI runners will retry to download Boost when necessary #1640
- Updated used CMake versions to their latest point releases #1638 #1649

## [0.8.0] - 2021-12-20
### Compatibility Changes:
- Platform support added:
  - clang 12 #1385
  - CUDA 11.4 #1380
  - GCC 11 #1383
  - HIP-clang #1338
  - Xcode 12.5.1 #1385
  - Xcode 13 #1421
- Platform support removed:
  - clang < 5.0 #1385
  - CUDA < 9.2 #1385
  - GCC < 7.0 #1385
  - HIP-nvcc #1337
  - Visual Studio < 2019 #1385
  - Ubuntu 16.04 #1352
  - Xcode 11.x < 11.3.1 #1385
  - Xcode 12.x < 12.4 #1385
 
### Bug Fixes:
- Added missing `#include <limits>` in a few places which would lead to compilation errors for CPU back-ends #1327
- Added missing `std::` to fixed-width integers where necessary #1327
- Fixed behavior of `assert` and `printf` for OpenMP offloading targets #1351
- `ALPAKA_STATIC_ACC_MEM_CONSTANT` now works correctly for clang-CUDA as well as HIP #1386
- The OpenMP 5 and OpenACC back-ends now correctly pass the parameters as an `is_trivially_copyable` type #1387
- Fixed `alpaka_compiler_option` checking for the wrong variable name #1392
- Fixed a bug in the HIP back-end's peer-to-peer `memcpy` implementation #1400
- The CMake function `alpaka_compiler_option` has been turned into a macro which solves parameter scope issues #1401
- Fixed compilation error occuring with CUDA >= 11.3 #1404
- The `-pthread` flag is now correctly passed to the (host) compiler and linker #1420
- alpaka now correctly sets the CUDA host compiler #1423
- alpaka's headers are now treated as CMake `SYSTEM` headers so that internal warnings no longer annoy users #1451
- Projects using alpaka can now set `ALPAKA_CXX_STANDARD` as variable in their `CMakeLists.txt` without alpaka ignoring this #1463

### New Features:
- alpaka now supports the Philox random number generator #1319
- alpaka's kernel language now supports memory fences (a.k.a thread fences) #1379
- `alpaka::Vec` now supports structured bindings #1393
- The OpenMP 5 and OpenACC back-ends now support statically mapped memory #1394
- alpaka now has factory methods for creating memory views #1398
- If OpenMP >= 5.1 is supported the back-end makes use of `atomic capture compare` #1411
- alpaka now experimentally supports accessors instead of pointers to access memory #1433
- Added wrapper for CUDA's native vector types so that they may be handled like arrays #1435
- Added function for **exact** floating point comparisons #1440
- Added portable implementations of random number distributions (default: `TinyMersenneTwister`) #1444
- Added new math functions: `isnan`, `isinf`, `isfinite` #1446
- New type trait that removes `__restrict__` from pointers #1474

### Misc:
- If TBB is enabled CMake is now able to pick up both oneAPI TBB and legacy TBB #1329
- Eclipse project files are no longer tracked by git #1347
- OpenACC atomics are now well-defined #1358
- Headers in `alpaka/test` are now installable #1360
- HIP no longer receives special treatment inside `alpaka_add_library` #1410
- Removed unnecessary annotations on default constructors #1416
- Removed unnecessary defaulted and deleted special member functions #1418
- Removed unnecessary `explicit` specifiers #1419
- Simplified implementation of `ALPAKA_UNROLL` #1437
- Math traits have been (internally) simplified #1457
- Visual Studio project files are no longer tracked by git #1464
- The alpaka CMake project now enables the `CXX` language by default everywhere (previously some test cases would enable `C`) #1470

### Breaking Changes:
- Legacy TBB support is deprecated. alpaka will move to oneTBB in 0.9 #1329
- HIP + nvcc is no longer supported #1337
- The behavior of `ALPAKA_STATIC_ACC_MEM_CONSTANT` and `ALPAKA_STATIC_ACC_MEM_GLOBAL` was changed #1386
- `alpaka::rand::engine::createDefault()` now features an additional `offset` parameter #1434

### Test cases / CI:
- NVHPC is now tested #1308
- Disabled MSVC + CUDA 11.{0,1} runners due to a CUDA bug #1332
- Some runtime tests are offloaded to HZDR's own CI #1375
- `install_clang.sh` now installs the correct versions of `libc++` and `libc++abi` #1385
- Fixed overflow in `AccDevPropsTest` #1395
- Fixed a bug in the HIP peer-to-peer `memcpy` test #1399
- Math tests no longer fail for clang-cuda #1406
- CUDA/HIP test cases are in part tested by HZDR's own CI #1407, #1409
- CI now uses `clang-format` 12.0.1 #1417, #1430
- atomic tests now also test `float` and `double` #1431
- all test executables have been renamed: `executable` is now called `executableTest` #1432
- test infrastructure is now based on Catch2 v2.13.7 #1461
- alpaka and all subprojects now only enable `CXX` by default #1473
- Removed unnecessary disabling of MySQL #1524
- Reflect HZDR GitLab CI node changes for HIP #1530


## [0.7.0] - 2021-08-03
### Compatibility Changes:
- Visual Studio 2017 is no longer supported #1251
- 32bit Windows is no longer supported #1251
- CUDA 11.3 is now supported #1295
- clang < 9 is no longer supported as CUDA compiler #1300
- clang 11 is now supported #1310

### Bug Fixes:
- fixed `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED` being checked without being defined #1259

### New Features:
- when no specialization is provided by the user alpaka's math functions will now fall back to ADL to find a candidate #1248
- the HIP back-end now supports callbacks #1269
- added warp::shfl functionality #1273
- added `Front` and `Contains` type list meta functions #1306

### Misc:
- alpaka's CMake build system now uses CMake's first-class CUDA support #1146
- updated documentation for clang-format usage #1222
- increased the static shared memory size to 47 KiB #1247
- fixed table markup in README.md #1256
- added example showcasing how to specialize kernels for particular back-ends #1271
- removed section comments #1275
- updated cheatsheet (added warp info, fixed names) #1281

### Breaking Changes:
- alpaka now requires CMake 3.18 or newer #1146
- the CUDA and HIP back-ends no longer enable fast-math by default #1285
- the CMake options `ALPAKA_CUDA_FAST_MATH` and `ALPAKA_HIP_FAST_MATH` have been replaced by `ALPAKA_FAST_MATH` #1289
- the CMake options `ALPAKA_CUDA_FTZ` and `ALPAKA_HIP_FTZ` have been replaced by `ALPAKA_FTZ` #1289
- the CMake option `ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION` has been replaced by the native CMake property `CUDA_SEPARABLE_COMPILATION` #1289
- the CMake option `ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA` has been replaced by `ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA` #1289

### Test cases / CI:
- enabled OpenMP back-ends for more Visual Studio builds #1219
- fixed gh-pages #1230
- added ICPC / ICC 2021.x to CI #1235
- fixed deadlock in Ubuntu 20.04 container #1270
- now CI-testing CMake 3.20 #1283

## [0.6.1] - 2021-06-29
### Compatibility Changes:
- rework implementation of OpenMP schedule support #1279 #1309 #1313 #1341
  - `alpaka::omp::Schedule` is replaced by `ompScheduleKind` and `ompScheduleChunkSize`
### Bug Fixes:
- fix OpenMP 5 shared memory allocation #1254 
- fix static shared memory alignment #1282
- fix BlockSharedMemStMemberImpl::getVarPtr for last var #1280
- fix CPU static shared memory implementation #1258
- unit tests: fix queue test #1266
- fix CtxBlockOacc: SyncBlockThreads #1291
- fix assert in DeclareSharedVar (OpenAcc) #1303
- CMake CUDA: dev compile options not propagated #1294
- example: fix warning (NVCC+OpenMP) #1307
- TBB: Add missing <limits> header and fix integer namespace #1327
- OpenAcc: TaskKernelOacc: copyin(all used local vars) #1342
- port macOSX CI fix from #1283
- CI: use ubuntu-18.04 for gcc-5 and gcc-6 builds #1252
- CI: disable GCC 10.3 + NVCC tests #1302
- CI: MSVC + nvcc workarounds and fixes #1332
- CI: fix warp test #1339

### Misc
- add ALPAKA_ASSERT_OFFLOAD Macro #1260
- document return value of `empty()` and `isComplete()` #1265
- Prefer TBBConfig.cmake over FindTBB.cmake #1329

## [0.6.0] - 2021-01-20
### Compatibility Changes:
- support for CUDA 11, 11.1, and 11.2 #1076 #1086 #1147 #1231
- remove support for CUDA 11.0 with MSVC 2019 #1227
- support for CMake 3.18.0 and 3.19.0 #1087 #1217
- set minimal HIP version to 3.5 #1110
- remove CMake HIP module shipped with alpaka #1189
- set HIP-clang as default compiler for HIP #1113
- support for NVCC + VS 2019 #1121
- support for boost-1.74.0 #1142
- explicitly require backends and do not enable them by default #1111
- remove support for Xcode 11.1 #1206 
- support Xcode 11.21 - 12.2.0 #1206
- update to Catch 2.13.3 #1215

### Bug Fixes:
- apply some clang-tidy fixes #1044
- fix CUDA/HIP accelerator concept usage #1064
- fix Intel compiler detection #1070
- CMake: build type CXX flag not passed to nvcc #1073
- work around Intel ICE (Internal Compiler Error) when using std::decay on empty template parameter packs #1074
- BoostPredef.hpp: Add redefinition of BOOST_COMP_PGI #1082
- fix min/max return type deduction #1085
- CMake: fix boost fiber linking #1088
- fix HIP-clang compile #1107
- fix CUDA/HIP cmake flags #1152
- fix error handling CUDA/HIP #1108
- ALPAKA_DECAY_T: Fix Intel detection, Add PGI #1116
- fix how to set HIP target architecture #1112 
- fix and improve block shared mem st member sanity checks #1128
- HIP: remove copy device2device workaround #1188
- pass native pointers to kernel instead of buffer objects #1193
- fix bug in `isPinned()` and `pin()` #1196
- fix marking of unit tests for concepts #1226

### New Features:
- add functions `alpaka::atomicAnd` et. al. as shortcuts to `alpaka::atomicOp<alpaka::AtomicAnd>` et. al. #1005
- warp voting functions #1003 #1049 #1090 #1092
- Sphinx Doc: Fix Doxygen integration on readthedocs #1042 #1093 #1151
- add cheat sheet to the docs #1057 #1177
- extend AccDevProps with shared memory size per block #1084
- OpenMP 5 target offload backend #1126
- OpenACC backend #1127
- option to set OpenMP schedule for the Omp2Blocks backend #1223

### Misc
- tests for BufferSlicing #1024
- use std::invoke_result_t instead of std::result_of_t when available #1047
- simplify shared memory usage in tests #1075 
- remove boost::aligned_alloc #1094
- add unit tests for work div #1095
- change examples (except reduce) to use getValidWorkDiv #1104
- example monte-carlo-integration #1106 
- invoke docker run only once instead of twice #1109
- cpu/SysInfo.hpp: Add #else for cpuid; Add PGI #1119
- Pgi std atomic workaround #1120
- make BlockSharedMemDynMember::staticAllocBytes a function #1118
- add IntrinsicFallback: basic fallback implementations #1122
- allow ALPAKA_CXX_STANDARD to propagate to nvcc with MSVC 1920 and above #1130
- add set kernel #1132
- make Queue test generic to handle QueueGenericThreads* with different devices #1133
- IdxBtOmp: Add GetIdx specialization for 1d #1140
- test CMAKE_CXX_EXTENSIONS=OFF #1153
- change block memory size back to be stored as 32 bit #1187
- add comments to math function traits that explain valid argument range #1190
- provide docker_retry #1191
- add .clang-format file #1204
- add CI check whether code is correctly formatted #1213
- make test/common a CMake INTERFACE library #1228

### Breaking changes:

The namespace structure of *alpaka* is now flattened. 
The [script](https://gist.github.com/sliwowitz/0a55e1bed6350f7fcae17ef0d430040d) can help you to apply the changes to your code.
The script only works if you used the full namespace `alpaka::*` for alpaka functions.

- removed namespace `alpaka::dev`
- removed namespace `alpaka::pltf`
- renamed function `alpaka::vec::cast` to `alpaka::castVec`
- renamed function `alpaka::vec::reverse` to `alpaka::reverseVec`
- renamed function `alpaka::vec::concat` to `alpaka::concatVec`
- removed namespace `alpaka::vec`
- removed namespace `alpaka::workdiv`
- removed namespace `alpaka::acc`
- renamed functors `alpaka::atomic::op::And` et. al. to `alpaka::AtomicAnd` et. al. #1185
- removed namespace `alpaka::atomic::op`
- removed namespace `alpaka::atomic`
- removed namespace `alpaka::queue`
- removed namespace `alpaka::idx`
- removed namespace `alpaka::dim`
- removed namespace `alpaka::kernel`
- removed namespace `alpaka::wait`
- removed namespace `alpaka::mem`
- removed namespace `alpaka::offset`
- removed namespace `alpaka::elem`
- removed namespace `alpaka::intrinsic`
- renamed function `alpaka::event::test` to `alpaka::isComplete`
- removed namespace `alpaka::event`
- removed namespace `alpaka::time`
- removed namespace `alpaka::example`
- renamed function `alpaka::alloc::alloc` to `alpaka::malloc`
- renamed function `alpaka::buf::alloc` to `alpaka::allocBuf`
- removed namespace `alpaka::alloc`
- removed namespace `alpaka::buf`
- renamed function `alpaka::view::set` to `alpaka::memset`
- renamed function `alpaka::view::copy` to `alpaka::memcpy`
- removed namespace `alpaka::view`
- removed namespace `alpaka::block::shared::st`
- removed namespace `alpaka::block::shared::dyn`
- removed namespace `alpaka::block::sync`
- renamed function `getMem` to `getDynSharedMem` #1197
- renamed function `getVar` to `declareSharedVar` #1197
- renamed function `freeMem` to `freeSharedVars` #1197
- renamed functors `alpaka::block::op::LogicalAnd` et. al. to `alpaka::BlockAnd` et. al.
- removed namespace `alpaka::block::op`
- removed namespace `alpaka::block`


## [0.5.0] - 2020-06-26
### Compatibility Changes:
- the minimum required C++ version has been raised from C++11 to C++14 #900
- drop support for CUDA 8.0 (does not support c++14)
- drop support for gcc 4.9 (does not support c++14)
- drop support for CMake versions lower than 3.15 (3.11, 3.12, 3.13 and 3.14)
- raise minimum supported boost version from 1.62.0 to 1.65.1 #906
- require HIP version to 3.3.0 #1006
- drop HIP-hcc support #945

### Bug Fixes:
- fix CMake error #941
- fix HIP math includes #947
- fix: missing hipRand and rocRand library #948
- fix VS 2017 CUDA builds #953
- fix uninitialized pitch #963
- fix windows CI builds #965
- fix conversion warning in TinyMT #997

### New Features:
- add automated gh-pages deployment for branch develop #916
- unify CUDA/HIP backend #928 #904 #950 #980 #981
- add support for Visual Studio 2019 #949
- simplify vector operator construction #977
- example heat-equation #978
- extend supported compiler combinations gcc-8+nvcc 10.1-10.2 #985
- add support for CMake 3.17 #988
- adds initial files for sphinx/rst and readthedocs. #990 #1017 #1048
- add support for clang 10 #998
- add popcount intrinsic #1004
- emulate hip/cuda-Memcpy3D with a kernel #1014
- simplify alpaka usage #1017


## [0.4.0] - 2020-01-14
### Compatibility Changes:
- added support for CUDA 10.0, 10.1 and 10.2
- dropped support for CUDA 7.0 and 7.5
- added official support for Visual Studio 2017 on Windows with CUDA 10 (built on Travis CI instead of appveyor now)
- added support for xcode10.2-11.3 (no official CUDA support yet)
- added support for Ubuntu 18.04
- added support for gcc 9
- added support for clang 7.0, 8.0 and 9.0
- dropped support for clang 3.5, 3.6, 3.7, 3.8 and 3.9
- added support for CMake 3.13, 3.14, 3.15 and 3.16
- dropped support for CMake 3.11.3 and lower, 3.11.4 is the lowest supported version
- added support for Boost 1.69, 1.70 and 1.71
- added support for usage of libc++ instead of libstdc++ for clang builds
- removed dependency to Boost.MPL and BOOST_CURRENT_FUNCTION
- replaced Boost.Test with Catch2 using an internal version of Catch2 by default but allowing to use an external one

### Bug Fixes:
- fixed some incorrect host/device function attributes
- fixed warning about comparison unsigned < 0
- There is no need to disable all other backends manually when using ALPAKA_ACC_GPU_CUDA_ONLY_MODE anymore
- fixed static block shared memory of types with alignemnt higher than defaultAlignment
- fixed race-condition in HIP/NVCC queue
- fixed data races when a GPU updates host memory by aligning host memory buffers always to 4kib

### New Features:
- Added a new alpaka Logo!
- the whole alpaka code has been relicensed to MPL2 and the examples to ISC
- added ALPAKA_CXX_STANDARD CMake option which allows to select the C++ standard to be used
- added ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION option to enable separable compilation for nvcc
- added ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA and ALPAKA_CUDA_NVCC_EXPT_RELAXED_CONSTEXPR CMake options to enable/disable those nvcc options (they were always ON before)
- added headers for standalone usage without CMake (alpaka/standalone/GpuCudaRt.h, ...) which set the backend defines
- added experimental HIP back-end with using nvcc (HIP >= 1.5.1 required, latest rocRand). More on HIP setup: doc/markdown/user/implementation/mapping/HIP.md
- added sincos math function implementations
- allowed to copy and move construct ViewPlainPtr
- added support for CUDA atomics using "unsigned long int"
- added compile-time error for atomic CUDA ops which are not available due to sm restrictions
- added explicit errors for unsupported types/operations for CUDA atomics
- replaced usages of assert with ALPAKA_ASSERT
- replaced BOOST_VERIFY by ALPAKA_CHECK and returned success from all test kernels
- added alpaka::ignore_unused as replacement for boost::ignore_unused

### Breaking changes:
- renamed Queue*Async to Queue*NonBlocking and Queue*Sync to Queue*Blocking
- renamed alpaka::size::Size to alpaka::idx::Idx, alpaka::size::SizeType to alpaka::idx::IdxType (and TSize to TIdx internally)
- replaced ALPAKA_FN_ACC_NO_CUDA by ALPAKA_FN_HOST
- replaced ALPAKA_FN_ACC_CUDA_ONLY by direct usage of __device__
- renamed ALPAKA_STATIC_DEV_MEM_CONSTANT to ALPAKA_STATIC_ACC_MEM_CONSTANT and ALPAKA_STATIC_DEV_MEM_GLOBAL to ALPAKA_STATIC_ACC_MEM_GLOBAL
- renamed alpaka::kernel::createTaskExec to alpaka::kernel::createTaskKernel
- QueueCpuSync now correctly blocks when called from multiple threads
  - This broke some previous use-cases (e.g. usage within existing OpenMP parallel regions)
  - This use case can now be handled with the support for external CPU queues as can bee seen in the example QueueCpuOmp2CollectiveImpl
- previously it was possible to have kernels return values even though they were always ignored. Now kernels are checked to always return void
- renamed all files with *Stl suffix to *StdLib
- renamed BOOST_ARCH_CUDA_DEVICE to BOOST_ARCH_PTX
- executors have been renamed due to the upcoming standard C++ feature with a different meaning. All files within alpaka/exec/ have been moved to alpaka/kernel/ and the files and classes have been renamed from Exec* to TaskKernel*. This should not affect users of alpaka but will affect extensions.

## [0.3.6] - 2020-01-06
### Bug Fixes:
- fix cuda stream race condition #850
- fix: cuda exceptions #844
- math/abs: Added trait specialisation for double. #862
- alpaka/math Overloaded float specialization #837
- Fixes name conflicts in alpaka math functions. #784


## [0.3.5] - 2018-11-18
### New Features:
- used OpenMP atomics instead of critical sections


## [0.3.4] - 2018-10-17
### Compatibility Changes:
- added support for boost-1.68.0
- added support for CUDA 10
- support for glibc < 2.18 (fix missing macros)
- added checks for available OpenMP versions

### Bug Fixes:
- fixed empty(StreamCpuAsync) returning true even though the last task is still in progress
- fixed integer overflows in case of int16_t being used as accelerator index type
- made some throwing destructors not throwing to support clang 7
- fixed broken alpaka::math::min for non-integral types

### New Features:
- added prepareForAsyncCopy which can be called to enable async copies for a specific buffer (if it is supported)
- allowed to run alpaka OpenMP 2 block accelerated kernels within existing parallel region
- added alpaka::ignore_unused which can be used in kernels


## [0.3.3] - 2018-08-10
### New Features:
- added CPU random number generators based on std::random_device and TinyMT32
- made TinyMT32 the default random number generator
- added alpaka::ignore_unused


## [0.3.2] - 2018-10-17
### New Features:
- Enhanced the compiler compatibility checks within the CMake scripts

### Bugs Fixed:
- fixed missing error in case of wrong OpenMP thread count being used by the runtime that was not triggered when not in debug mode
- fixed CUDA driver API error handling
- fixed CUDA memcpy and memset for zero sized buffers (division by zero)
- fixed OpenMP 4 execution
- fixed the VS2017 CUDA build (not officially supported)
- fixed CUDA callback execution not waiting for the task to finish executing
- fixed cudaOnly test being part of make test when cuda only mode is not enabled

### Compatibility Changes:
- added support for CUDA 9.2


## [0.3.1] - 2018-06-11
### New Features:
- CMake: added option to control tests BUILD_TESTING
- CMake: unified requirement of CMake 3.7.0+
- CMake: used targets for Boost dependencies
- CMake: made alpaka a pure interface library

### Bugs Fixed:
- fixed getDevCount documentation
- fixed undefined define warnings
- fixed self containing header check for CUDA


## [0.3.0] - 2018-03-15
### Bugs Fixed:
- fixed multiple bugs where CPU streams/events could deadlock or behaved different than the native CUDA events
- fixed a bug where the block synchronization of the Boost.Fiber backend crashed due to uninitialized variables

### New Features / Enhancements:
- added support for stream callbacks allowing to enqueue arbitrary host code using alpaka::stream::enqueue(stream, [&](){...});
- added support for compiling for multiple architectures using e.g. ALPAKA_CUDA_ARCH="20;35"
- added support for using __host__ constexpr code within __device__ code
- enhanced the CUDA error handling
- enhanced the documentation for mapping CUDA to alpaka

### Compatibility Changes:
- added support for CUDA 9.0 and 9.1
- added support for CMake 3.9 and 3.10
- removed support for CMake 3.6 and older
- added support for boost-1.65.0
- removed support for boost-1.61.0 and older
- added support for gcc 7
- added support for clang 4 and 5
- removed support for VS2015


## [0.2.0] - 2017-06-19
### Compatibility fixes and small enhancements:
- the documentation has been greatly enhanced
- adds support for CUDA 8.0
- adds support for CMake versions 3.6, 3.7 and 3.8
- adds support for Boost 1.62, 1.63 and 1.64
- adds support for clang-3.9
- adds support for Visual Studio 2017
- alpaka now compiles clean even with clang -Weverything
- re-enabled the boost::fiber accelerator backend which was disabled in the last release

### API changes:
- mapIdx is moved from namespace alpaka::core to alpaka::idx
- Vec is moved from namespace alpaka to alpaka::vec
- vec::Vec is now allowed to be zero-dimensional (was previously forbidden)
- added vec::concat
- added element-wise operator< for vec::Vec which returns a vector of bool
- CPU accelerators now support arbitrary dimensionality (both kernel execution as well as memory operations)
- added support for syncBlockThreadsPredicate with block::sync::op::LogicalOr, block::sync::op::LogicalAnd and block::sync::op::Count
- memory allocations are now aligned optimally for the underlying architecture (16 bit for SSE, 32 bit for AVX, 64 bit for AVX512) instead of 16 bit for all architectures in the previous release
