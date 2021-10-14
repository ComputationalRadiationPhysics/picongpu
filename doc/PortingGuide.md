Requirements to Port Your Project to *cupla*
============================================

- your build system must be `CMake`
- your code must be compatible with C++14


Reserved Variable Names
=======================

Some variable names are forbidden to use on the host side and are only allowed
in kernels:
  - `blockDim`
  - `gridDim`
  - `elemDim` number of elements per thread (is a three dimensional struct)
  - `blockIdx`
  - `threadIdx`


Restrictions
============

Cupla host-side API is not thread-safe.
Thus, it is not allowed to use cupla API from multiple host-side threads simultaneously.
In such scenarios, some synchronization between the host-side threads is required to ensure safe usage of cupla.

Events with timing information synchronize the stream where they were recorded.
Disable the timing information of the event by setting the flag
`cudaEventDisableTiming` or `cuplaEventDisableTiming` during the event
creation.


Porting Step by Step
====================

- change the suffix `*.cu` of the CUDA source files to `*.cpp`
- remove cuda specific includes on top of your header and source files
- add include `cuda_to_cupla.hpp`

  **CUDA include**
  ```C++
  #include <cuda_runtime.h>
  ```

  **cupla include**
  ```C++
  /* Do NOT include other headers that use CUDA runtime functions or variables
   * (see above) after this include.
   * In particular, this applies to third-party library headers pulling CUDA -
   * they also have to be included before cupla.
   * The reason for this is that cupla renames CUDA host functions and device build in
   * variables by using macros and macro functions.
   * Do NOT include other specific includes such as `<cuda.h>` (driver functions,
   * etc.).
   */
  #include <cuda_to_cupla.hpp>
  ```

- transform kernels (`__global__` functions) to functors
- the functor's `operator()` must be qualified as `const`
- add the function prefix `ALPAKA_FN_ACC` to the `operator() const`
- add as first (templated) kernel parameter the accelerator with the name `acc`
  (it is important that the accelerator is named `acc` because all
  cupla-to-alpaka replacements use the variable `acc`)
- if the kernel calls other functions you must pass the accelerator `acc`
  to each call
- add the qualifier `const` to each parameter which is not changed inside the
  kernel

  **CUDA kernel**
  ```C++
  template< int blockSize >
  __global__ void fooKernel( int * ptr, float value )
  {
      // ...
  }
  ```

  **cupla kernel**
  ```C++
  template< int blockSize >
  struct fooKernel
  {
      template< typename T_Acc >
      ALPAKA_FN_ACC
      void operator()( T_Acc const & acc, int * const ptr, float const value) const
      {
          // ...
      }
  };
  ```

- The host side kernel call must be changed like this:

  **CUDA host side kernel call**
  ```C++
  // ...
  dim3 gridSize(42,1,1);
  dim3 blockSize(256,1,1);
  // extern shared memory and stream is optional
  fooKernel< 16 ><<< gridSize, blockSize, 0, 0 >>>( ptr, 23 );
  ```

  **cupla host side kernel call**
  ```C++
  // ...
  dim3 gridSize(42,1,1);
  dim3 blockSize(256,1,1);
  // extern shared memory and stream is optional
  CUPLA_KERNEL(fooKernel< 16 >)( gridSize, blockSize, 0, 0 )( ptr, 23 );
  ```

- Static shared memory definitions

  **Cuda shared memory** (in kernel)
  ```C++
  // ...
  __shared__ int foo;
  __shared__ int fooCArray[32];
  __shared__ int fooCArray2D[4][32];

  // extern shared memory (size was defined during the host side kernel call)
  extern __shared__ float fooPtr[];

  int bar = fooCArray2D[ threadIdx.x ][ 0 ];
  // ...
  ```

  **cupla shared memory** (in kernel)
  ```C++
  // ...
  sharedMem( foo, int );
  /* It is not possible to use the C-notation of fixed size, shared memory
   * C arrays in cupla. Instead use `cupla::Array<Type,size>`.
   */
  sharedMem( fooCArray, cupla::Array< int, 32 > );
  sharedMem( fooCArray, cupla::Array< cupla::Array< int, 4 >, 32 > );

  /* extern shared memory (size was defined during the host side kernel call)
   *
   * The type of extern shared memory is always a plain pointer of the given type.
   * In this example the type of `fooPtr` is `float*`
   */
  sharedMemExtern( fooPtr, float );

  int bar = fooCArray2D[ threadIdx.x ][ 0 ];
  // ...
  ```

- use `ALPAKA_FN_ACC` in device function definitions and add an `acc` parameter. Note that to be exact the `acc` parameter is only necessary when `alpaka` functions like `blockIdx` or `atomicMax`, ... are used.

  **CUDA**

      template< typename T_Elem >
      __device__ int deviceFunction( T_Elem x )
      {
          // ...
      }

  **cupla**

      template< typename T_Acc, typename T_Elem >
      ALPAKA_FN_ACC int deviceFunction( T_Acc const & acc, T_Elem x )
      {
          // ...
      }

- add `acc` as first parameter to device function calls

   **CUDA**

       auto result = deviceFunction( x );

   **cupla**

       auto result = deviceFunction( acc, x );

- Cupla code can be mixed with
  [**alpaka**](https://github.com/alpaka-group/alpaka)
  low level code. This becomes necessary as you are progressing to write more
  general, performance portable code. Additional functionality provided by
  alpaka includes for example, platform independent math functions inside
  kernels, has lower runtime overhead for some cupla runtime functions and
  is type save (buffer objects instead of `void *` pointers).
