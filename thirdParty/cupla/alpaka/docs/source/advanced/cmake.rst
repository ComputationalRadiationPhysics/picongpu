CMake Arguments
===============

Alpaka configures a lot of its functionality at compile time. Therefore a lot of compiler and link flags are needed, which are set by ``CMake`` arguments. The beginning of this section introduces the general Alpaca flag. The last parts of the section describe back-end specific flags.

.. hint::

   To display the cmake variables with value and type in the build folder of your project, use ``cmake -LH <path-to-build>``.

**Table of back-ends**

   * :ref:`CPU Serial <cpu-serial>`
   * :ref:`C++ Threads <cpp-threads>`
   * :ref:`Boost Fiber <boost-fiber>`
   * :ref:`Intel TBB <intel-tbb>`
   * :ref:`OpenMP 2 Grid Block <openmp2-grid-block>`
   * :ref:`OpenMP 2 Block Thread <openmp2-block-thread>`
   * :ref:`OpenMP 5 <openmp5>`
   * :ref:`CUDA <cuda>`
   * :ref:`HIP <hip>`

Common
------

ALPAKA_CXX_STANDARD
  .. code-block::

     Set the C++ standard version.

alpaka_BUILD_EXAMPLES
  .. code-block::

     Build the examples.

BUILD_TESTING
  .. code-block::

     Build the testing tree.

ALPAKA_DEBUG
  .. code-block::

     Set Debug level:

     0 - Is the default value. No additional logging.
     1 - Enables some basic flow traces.
     2 - Display as many information as possible. Especially pointers, sizes and other
         parameters of copies, kernel invocations and other operations will be printed.

ALPAKA_USE_INTERNAL_CATCH2
  .. code-block::

     Use internally shipped Catch2.


ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST
  .. code-block::

     Allow host-only contructs like assert in offload code in debug mode.

.. _cpu-serial:

CPU Serial
----------

ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE
  .. code-block::

     Enable the serial CPU back-end.

ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _cpp-threads:

C++ Threads
-----------

ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE
  .. code-block::

     Enable the threads CPU block thread back-end.

.. _boost-fiber:

Boost Fiber
-----------

ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE
  .. code-block::

     Enable the fibers CPU block thread back-end.

.. _intel-tbb:

Intel TBB
---------

ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE
  .. code-block::

     Enable the TBB CPU grid block back-end.

ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _openmp2-grid-block:

OpenMP 2 Grid Block
-------------------

ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE
  .. code-block::

     Enable the OpenMP 2.0 CPU grid block back-end.

ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _openmp2-block-thread:

OpenMP 2 Block thread
---------------------

ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE
  .. code-block::

     Enable the OpenMP 2.0 CPU block thread back-end.

.. _openmp5:

OpenMP 5
--------

ALPAKA_ACC_ANY_BT_OMP5_ENABLE
  .. code-block::

     Enable the OpenMP 5.0 CPU block and block thread back-end.


ALPAKA_OFFLOAD_MAX_BLOCK_SIZE
  .. code-block::

     Maximum number threads per block to be suggested by any target offloading backends
     ANY_BT_OMP5 and ANY_BT_OACC.

.. _cuda:

CUDA
----

ALPAKA_ACC_GPU_CUDA_ENABLE
  .. code-block::

     Enable the CUDA GPU back-end.

ALPAKA_ACC_GPU_CUDA_ONLY_MODE
  .. code-block::

     Only back-ends using CUDA can be enabled in this mode (This allows to mix
     alpaka code with native CUDA code).


ALPAKA_CUDA_ARCH
  .. code-block::

     Set the GPU architecture: e.g. "35".

ALPAKA_CUDA_COMPILER
  .. code-block::

     Set the CUDA compiler: "nvcc" or "clang".

ALPAKA_CUDA_FAST_MATH
  .. code-block::

     Enable fast-math.

ALPAKA_CUDA_FTZ
  .. code-block::

     Set flush to zero for GPU.

ALPAKA_CUDA_KEEP_FILES
  .. code-block::

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA
  .. code-block::

     Enable experimental, extended host-device lambdas in NVCC.

ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION
  .. code-block::

     Enable separable compilation in NVCC.

https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

ALPAKA_CUDA_SHOW_CODELINES
  .. code-block::

     Show kernel lines in cuda-gdb and cuda-memcheck. If ALPAKA_CUDA_KEEP_FILES
     is enabled source code will be inlined in ptx.
     One of the added flags is: --generate-line-info

ALPAKA_CUDA_SHOW_REGISTER
  .. code-block::

     Show the number of used kernel registers during compilation and create PTX.

.. _hip:

HIP
---

To enable the HIP backend please provide the path to the CMake find module `FindHIP.cmake`.
The path can be given via an environment variable `CMAKE_MODULE_PATH` or by providing the CMake flag `-DCMAKE_MODULE_PATH=<path>`.

ALPAKA_ACC_GPU_HIP_ENABLE
  .. code-block::

     Enable the HIP back-end (all other back-ends must be disabled).

ALPAKA_ACC_GPU_HIP_ONLY_MODE
  .. code-block::

     Only back-ends using HIP can be enabled in this mode.

ALPAKA_HIP_PLATFORM
  .. code-block::

     Specify HIP platform. Can be "clang" or "nvcc".

ALPAKA_HIP_KEEP_FILES
  .. code-block::

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.
