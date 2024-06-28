CMake Arguments
===============

Alpaka configures a large part of its functionality at compile time. Therefore, a lot of compiler and link flags are needed, which are set by CMake arguments. First, we show a simple way to build alpaka for different back-ends using `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_. The second part of the documentation shows the general and back-end specific alpaka CMake flags.

.. hint::

   To display the cmake variables with value and type in the build folder of your project, use ``cmake -LH <path-to-build>``.

CMake Presets
-------------

The `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_ are defined in the ``CMakePresets.json`` file. Each preset contains a set of CMake arguments. We use different presets to build the examples and tests with different back-ends. Execute the following command to display all presets:

.. code-block:: bash

   cd <alpaka_project_root>
   cmake --list-presets

To configure, build and run the tests of a specific preset, run the following commands (for the example, we use the ``cpu-serial`` preset):

.. code-block:: bash

   cd <alpaka_project_root>
   # configure a specific preset
   cmake --preset cpu-serial
   # build the preset
   cmake --build --preset cpu-serial
   # run test of the preset
   ctest --preset cpu-serial

All presets are configure and build in a subfolder of the ``<alpaka_project_root>/build`` folder.

Modifying and Extending Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to change a preset is to set CMake arguments during configuration:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the cpu-serial preset with clang++ as C++ compiler
   cmake --preset cpu-serial -DCMAKE_CXX_COMPILER=clang++
   # build the preset
   cmake --build --preset cpu-serial
   # run test of the preset
   ctest --preset cpu-serial

It is also possible to configure the default setting first and then change the arguments with ``ccmake``:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the cpu-serial preset with clang++ as C++ compiler
   cmake --preset cpu-serial
   cd build/cpu-serial
   ccmake .
   cd ../..
   # build the preset
   cmake --build --preset cpu-serial
   # run test of the preset
   ctest --preset cpu-serial

CMake presets also offer the option of creating personal, user-specific configurations based on the predefined CMake presets. To do this, you can create the file ``CMakeUserPresets.json`` in the root directory of your project (the file is located directly next to ``CMakePresets.json``). You can then create your own configurations from the existing CMake presets. The following example takes the cpu-serial configuration, uses ``ninja`` as the generator instead of the standard generator and uses the build type ``RELEASE``.

.. code-block:: json

   {
      "version": 3,
      "cmakeMinimumRequired": {
        "major": 3,
        "minor": 22,
        "patch": 0
      },
      "configurePresets": [
        {
            "name": "cpu-serial-ninja-release",
            "inherits": "cpu-serial",
            "generator": "Ninja",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": {
                    "type": "STRING",
                    "value": "RELEASE"
                }
            }
        }
      ]
   }

.. hint::

   Many IDEs like `Visual Studio Code <https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-presets.md>`_ and `CLion <https://www.jetbrains.com/help/clion/cmake-presets.html>`_ support CMake presets.

Arguments
---------

**Table of back-ends**

   * :ref:`CPU Serial <cpu-serial>`
   * :ref:`C++ Threads <cpp-threads>`
   * :ref:`Intel TBB <intel-tbb>`
   * :ref:`OpenMP 2 Grid Block <openmp2-grid-block>`
   * :ref:`OpenMP 2 Block Thread <openmp2-block-thread>`
   * :ref:`CUDA <cuda>`
   * :ref:`HIP <hip>`

Common
^^^^^^

alpaka_CXX_STANDARD
  .. code-block::

     Set the C++ standard version.

alpaka_BUILD_EXAMPLES
  .. code-block::

     Build the examples.

alpaka_BUILD_BENCHMARKS
  .. code-block::

     Build the benchmarks.

BUILD_TESTING
  .. code-block::

     Build the testing tree.

alpaka_INSTALL_TEST_HEADER
  .. code-block::

     Install headers of the namespace alpaka::test.
     Attention, headers are not designed for production code.
     They should only be used for prototyping or creating tests that use alpaka
     functionality.

alpaka_DEBUG
  .. code-block::

     Set Debug level:

     0 - Is the default value. No additional logging.
     1 - Enables some basic flow traces.
     2 - Display as many information as possible. Especially pointers, sizes and other
         parameters of copies, kernel invocations and other operations will be printed.

alpaka_USE_INTERNAL_CATCH2
  .. code-block::

     Use internally shipped Catch2.

alpaka_FAST_MATH
  .. code-block::

     Enable fast-math in kernels.

  .. warning::

     The default value is changed to "OFF" with alpaka 0.7.0.

alpaka_FTZ
  .. code-block::

     Set flush to zero for GPU.

alpaka_DEBUG_OFFLOAD_ASSUME_HOST
  .. code-block::

     Allow host-only constructs like assert in offload code in debug mode.

alpaka_USE_MDSPAN
  .. code-block::

     Enable/Disable the use of `std::experimental::mdspan`:

     "OFF" - Disable mdspan
     "SYSTEM" - Enable mdspan and acquire it via `find_package` from your system
     "FETCH" - Enable mdspan and download it via CMake's `FetchContent` from GitHub. The dependency will not be installed when you install alpaka.

.. _cpu-serial:

CPU Serial
^^^^^^^^^^

alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE
  .. code-block::

     Enable the serial CPU back-end.

alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _cpp-threads:

C++ Threads
^^^^^^^^^^^

alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE
  .. code-block::

     Enable the threads CPU block thread back-end.

.. _intel-tbb:

Intel TBB
^^^^^^^^^

alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE
  .. code-block::

     Enable the TBB CPU grid block back-end.

alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _openmp2-grid-block:

OpenMP 2 Grid Block
^^^^^^^^^^^^^^^^^^^

alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE
  .. code-block::

     Enable the OpenMP 2.0 CPU grid block back-end.

alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
  .. code-block::

     Kibibytes (1024B) of memory to allocate for block shared memory for backends
     requiring static allocation.

.. _openmp2-block-thread:

OpenMP 2 Block thread
^^^^^^^^^^^^^^^^^^^^^

alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE
  .. code-block::

     Enable the OpenMP 2.0 CPU block thread back-end.

.. _cuda:

CUDA
^^^^

alpaka_ACC_GPU_CUDA_ENABLE
  .. code-block::

     Enable the CUDA GPU back-end.

alpaka_ACC_GPU_CUDA_ONLY_MODE
  .. code-block::

     Only back-ends using CUDA can be enabled in this mode (This allows to mix
     alpaka code with native CUDA code).


CMAKE_CUDA_ARCHITECTURES
  .. code-block::

     Set the GPU architecture: e.g. "35;72".

CMAKE_CUDA_COMPILER
  .. code-block::

     Set the CUDA compiler: "nvcc" (default) or "clang++".

CUDACXX
  .. code-block::

     Select a specific CUDA compiler version.

alpaka_CUDA_KEEP_FILES
  .. code-block::

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

alpaka_CUDA_EXPT_EXTENDED_LAMBDA
  .. code-block::

     Enable experimental, extended host-device lambdas in NVCC.

alpaka_RELOCATABLE_DEVICE_CODE
  .. code-block::

     Enable relocatable device code. Note: This affects all targets in the
     CMake scope where ``alpaka_RELOCATABLE_DEVICE_CODE`` is set. For the
     effects on CUDA code see NVIDIA's blog post:

https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

alpaka_CUDA_SHOW_CODELINES
  .. code-block::

     Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES
     is enabled source code will be inlined in ptx.
     One of the added flags is: --generate-line-info

alpaka_CUDA_SHOW_REGISTER
  .. code-block::

     Show the number of used kernel registers during compilation and create PTX.

.. _hip:

HIP
^^^

To enable the HIP back-end please extend ``CMAKE_PREFIX_PATH`` with the path to the HIP installation.

alpaka_ACC_GPU_HIP_ENABLE
  .. code-block::

     Enable the HIP back-end (all other back-ends must be disabled).

alpaka_ACC_GPU_HIP_ONLY_MODE
  .. code-block::

     Only back-ends using HIP can be enabled in this mode.

CMAKE_HIP_ARCHITECTURES
  .. code-block::

     Set the GPU architecture: e.g. "gfx900;gfx906;gfx908".

A list of the GPU architectures can be found `here <https://llvm.org/docs/AMDGPUUsage.html#processors>`_.

alpaka_HIP_KEEP_FILES
  .. code-block::

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

alpaka_RELOCATABLE_DEVICE_CODE
  .. code-block::

     Enable relocatable device code. Note: This affects all targets in the
     CMake scope where ``alpaka_RELOCATABLE_DEVICE_CODE`` is set. For the
     effects on HIP code see the NVIDIA blog post linked below; HIP follows
     CUDA's behaviour.

https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

.. _sycl:

SYCL
^^^^

alpaka_RELOCATABLE_DEVICE_CODE
  .. code-block::

     Enable relocatable device code. Note: This affects all targets in the
     CMake scope where ``alpaka_RELOCATABLE_DEVICE_CODE`` is set. For the
     effects on SYCL code see Intel's documentation:

https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/fsycl-rdc.html
