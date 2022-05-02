# Building configuring for the OMP5 backend

```
cmake \
  -Dalpaka_ACC_ANY_BT_OMP5_ENABLE=on \
  -DBUILD_TESTING=on \
  -Dalpaka_BUILD_EXAMPLES=on \
```
Also set `-DCMAKE_CXX_FLAGS` in accordance with compiler and target. Information
about this is maintained
[on readthedocs](https://alpaka.readthedocs.io/en/latest/advanced/cmake.html).

## Block-shared Memory

Shared memory is implemented using a small object allocator in
`BlockSharedMemStOmp5` using a fixed-size buffer allocated by
`BlockSharedMemDynMember`, making these two elements linked.

OpenMP 5 offers the directive `omp allocate allocator(omp_pteam_mem_alloc)`
(used by `BlockSharedMemStOmp5BuiltIn`) which can in theory be used for *static*
shared memory variables. There is no useful built-in support for dynamic
block-shared memory to go with that. Usage of the built-in can be configured
using the `alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM` flag:
* `alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM=OFF`: Do not use `omp allocate` (default,
  only available behavior with OpenMP < 5).
* `alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM=DYN_FIXED`: Use `omp allocate`, use a
  fixed size team-shared array for dynamic shared mem (fixed size is
  `alpaka_BLOCK_SHARED_DYN_MEMBER_KIB`).
* `alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM=DYN_ALLOC`: Use `omp allocate`, use a
  `omp_alloc()` API call in the target region to allocate dynamic shared memory. The
  standard appears to allow this, but is not useful for some reasons:
  * In the best case, this would lead to an on-device `malloc` on GPU, which has
    bad performance and does not use on-chip memory.
  * At least in clang GPU targets (nvptx64, hsa), the symbols `omp_alloc` and `omp_free`
    are undefined (linker error, code compiles).

### Compiler support

[blockSharedSharingTest](test/unit/block/sharedSharing/src/BlockSharedMemSharing.cpp) tests correct sharing.

| compiler | target | `OFF` | `DYN_FIXED` | `DYN_ALLOC` |
| --- | --- | --- | --- | --- |
| clang 14 (1.) | x86 | :white_check_mark: | :white_check_mark: (2.) | :white_check_mark: (2.) |
| clang 14 (1.) | nvptx | :white_check_mark: | :white_check_mark: | E (3.) |
| clang 14 (1.) | hsa | C | C | E (3.) |
| gcc 11 | x86 | :white_check_mark: | N | N |
| gcc 11 | nvptx | :white_check_mark:/:x: (4.) | N | N |
| nvhpc 22.1 | x86 | :white_check_mark: | N (5.) | N (5.) |
| nvhpc 22.3 | nvptx | :white_check_mark: | N (5.) | N (5.) |

Keys:
* :white_check_mark:: Test Passes.
* :x:l: Test fails, shared mem not shared.
* :x:g: Test fails, shared mem gloal/shared too widely.
* :x:: Test fails for other reason.
* C: Test compiles, not run.
* E: Test does not build.
* N: Not supported.

Footnotes:
1. git main `95a436f8cca6991dc0f30588d9b1af3223818168`
2. `omp allocate` does not actually work, the variable being `static` makes it
   work, which in itself is non-conforming behavior.
3. Linker error: no symbols `omp_alloc`, `omp_free` for target code.
4. Apparently gcc's OpenMP runtime will not run more than 8 threads per block on
   GPU: Pass for `blockThreadCount <= 8`, fail for more.
5. NVHPC 22.1 claims to support OpenMP 5.1 (`_OPENMP = 202011`).

## Limitations

* *No separabel compilation*. OpenMP 5 requires functions for which device code should be generated for a
  not-inlined call in a target region to be marked with pragmas. This cannot be
  wrapped by macros like `ALPAKA_FN_DEVICE` because they appear between template
  parameter list and function name and because OpenMP requires two macros to
  mark a region around the function.
  <https://github.com/alpaka-group/alpaka/pull/1126#discussion_r479761867>

## 1. Test target

```
make vectorAdd
./example/vectorAdd/vectorAdd
```
If the run is successful, the last output line will be `Execution results
correct!` otherwise it will print items where the result from the offload code
disagrees with the expected result and print `Execution results
incorrect!` at the end.

## 2. Examples compilation status

### branch omp4

|target|compiler|compile status|target|run status|
|---|---|---|---|---|
|vectorAdd|
||GGC 10 | ok|host|ok|
||GGC 10 | ptxas error (2)|nvptx|--|
||AOMP 0.7-4|ok|x86|omp_target_alloc() returns 0|
||AOMP 0.7-4|linker: multiple def. of gpuHeap (1)|amdhsa|--|
||AOMP 0.7-5|ok|x86|ok|
||AOMP 0.7-5|ok	|amdhsa|ok|
||LLVM 10 |ok| x86|ok|
||XL 16.1.1-5 (Summit)|ok| nvptx|ok (num_threads workaround) (3)|
||XL 16.1.1-5 (Summit)|ok| ppc64le| sigsegv (device mem alloc'son GPU)|

#### errors:
1. error: Linking globals named 'gpuHeap': symbol multiply defined!
   ```
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
    clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see 
    invocation)
    clang-9: error: linker command failed with exit code 1 (use -v to see 
    invocation)
   ```
2. ptxas:
   ```
    Linking CXX executable vectorAdd
    ptxas /tmp/cccKHuiQ.o, line 216; error   : Label expected for argument 0 of instruction 'call'
    ptxas /tmp/cccKHuiQ.o, line 216; error   : Function '_ZN6alpaka3ctx12CtxBlockOaccISt17integral_constantImLm1EEmEC1ERKNS_3vec3VecIS3_mEES9_S9_RKmSB_' not declared in this scope
    ptxas /tmp/cccKHuiQ.o, line 216; fatal   : Call target not recognized
    ptxas fatal   : Ptx assembly aborted due to errors
    nvptx-as: ptxas returned 255 exit status
   ```
3. IBM XL: When setting num_threads, either in #pragma omp parallel or via
   omp_set_num_threads to any value the runtime only executes one thread per
   team. Workaround is to not do that with XL, which leads to $OMP_NUM_THREADS
   being run per team. Minimal example:
   https://github.com/jkelling/omp5tests/blob/master/parallel/parallel.cpp

## 3. Integration and Unit Tests

Run `make` and upon success `ctest`.

|test|compiler|compile status|target|run status|
|---|---|---|---|---|
|ALL|
||LLVM 10 |ok|x86|pass|
||LLVM 11 |ok|x86|pass|
||AOMP 0.7-5|linker error with static lib (7)x86|--|
||AOMP 0.7-5|linker error with static lib (8)|amdhsa|--|
||GCC 10 |mixed(1)|host|target alloc fail(2)|
||GCC 11 |ok|host|target alloc fail(2)|
||XL 16.1.1-5 (Summit)|no-halt [6]|nvptx|--|
||XL 16.1.1-5 (Summit)|no-halt [6]|ppc64le|--|

#### errors:
1. Targets with multiple compilation units fail to link.
   <https://github.com/alpaka-group/alpaka/pull/1126#discussion_r475591568>
2. `omp_target_alloc()` allocates memory on GPU while code runs on host and
   tries access it there => segfault
3. _
4. _
5. _
6. XL does not appear to terminate when compiling targets like `blockShared` in
   which tests are executed through the fixture in
   ~alpaka/test/common/include/alpaka/test/KernelExecutionFixture.hpp .
   Removing the call
   alpaka/test/unit/block/shared/src/BlockSharedMemDyn.cpp:92-94 yields finite
   compilation time for BlockSharedMemDyn.cpp.o . XL is extremely slow
   compiling code using the test framework catch2 used in Alpaka.
7. aomp 0.7-5 x86:
   ```
   /usr/bin/ld: cannot find libcommon-openmp-x86_64-pc-linux-gnu-sm_20.o: No such file or directory
   /usr/bin/ld: cannot find libcommon-host-x86_64-unknown-linux-gnu.o: No such file or directory
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   test/integ/matMul/CMakeFiles/matMul.dir/build.make:85: recipe for target 'test/integ/matMul/matMul' failed
   ```
8. aomp 0.7-5 HSA:
   ```
   /home/kelling/rocm/aomp_0.7-5/bin/clang-build-select-link: libcommon-openmp-amdgcn-amd-amdhsa-gfx900.o:1:2: error: expected integer
   !<arch>
    ^
   /home/kelling/rocm/aomp_0.7-5/bin/clang-build-select-link: error:  loading file 'libcommon-openmp-amdgcn-amd-amdhsa-gfx900.o'
   /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
   /usr/bin/ld: cannot find a.out-openmp-amdgcn-amd-amdhsa-gfx900
   clang-9: error: amdgcn-link command failed with exit code 1 (use -v to see invocation)
   clang-9: error: linker command failed with exit code 1 (use -v to see invocation)
   ```
