.. highlight:: bash

HIP Installation Notes
======================

Current restrictions on HCC platform
````````````````````````````````````

- Workaround for unsupported ``syncthreads_{count|and|or}``.
  - uses temporary shared value and atomics
- Workaround for buggy ``hipStreamQuery``, ``hipStreamSynchronize``.
  - introduces own queue management
  - ``hipStreamQuery`` and ``hipStreamSynchronize`` did not work in multithreaded environment
- Workaround for missing ``cuStreamWaitValue32``.
  - polls value each 10ms
- device constant memory not supported yet
- note, that ``printf`` in kernels is still not supported in HIP
- exclude ``hipMalloc3D`` and ``hipMallocPitch`` when size is zero otherwise they throw an Unknown Error
- ``TestAccs`` excludes 3D specialization of Hip back-end for now because ``verifyBytesSet`` fails in ``memView`` for 3D specialization
- ``dim3`` structure is not available on device (use ``alpaka::vec::Vec`` instead)
- Constructors' attributes unified with destructors'.
  - host/device signature must match in HIP(HCC)
- a chain of functions must also provide correct host-device signatures
  - e.g. a host function cannot be called from a host-device function
- recompile your target when HCC linker returned the error:
  "File format not recognized
  clang-7: error: linker command failed with exit code 1"
- if compile-error occurred, the linker still may link, but without the device code
- AMD device architecture currently hardcoded in ``alpakaConfig.cmake``

Compiling HIP from source
`````````````````````````

Follow `HIP Installation`_ guide for installing HIP.
HIP requires either *nvcc* or *hcc* to be installed on your system (see guide for further details).

.. _HIP Installation: https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md

- If you want the hip binaries to be located in a directory that does not require superuser access, be sure to change the install directory of HIP by modifying the ``CMAKE_INSTALL_PREFIX`` cmake variable.
- Also, after the installation is complete, add the following line to the ``.profile`` file in your home directory, in order to add the path to the HIP binaries to PATH: ``PATH=$PATH:<path_to_binaries>``

.. code-block::

   git clone --recursive https://github.com/ROCm-Developer-Tools/HIP.git
   cd HIP
   mkdir -p build
   cd build
   cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX=${YOUR_HIP_INSTALL_DIR} -DBUILD_TESTING=OFF ..
   make
   make install

- Set the appropriate paths (edit ``${YOUR_**}`` variables)

.. code-block::

  # HIP_PATH required by HIP tools
  export HIP_PATH=${YOUR_HIP_INSTALL_DIR}
  # Paths required by HIP tools
  export CUDA_PATH=${YOUR_CUDA_ROOT}
  # - if required, path to HCC compiler. Default /opt/rocm/hcc.
  export HCC_HOME=${YOUR_HCC_ROOT}
  # - if required, path to HSA include, lib. Default /opt/rocm/hsa.
  export HSA_PATH=${YOUR_HSA_PATH}
  # HIP binaries and libraries
  export PATH=${HIP_PATH}/bin:$PATH
  export LD_LIBRARY_PATH=${HIP_PATH}/lib64:${LD_LIBRARY_PATH}

- Test the HIP binaries

.. code-block::

  # calls nvcc or hcc
  which hipcc
  hipcc -V
  which hipconfig
  hipconfig -v


Verifying HIP installation
``````````````````````````

- If PATH points to the location of the HIP binaries, the following command should list several relevant environment variables, and also the selected compiler on your ``system-\`hipconfig -f\```
- Compile and run the `square sample`_, as pointed out in the original `HIP install guide`_.

.. _square sample: https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square
.. _HIP install guide: https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#user-content-verify-your-installation

Compiling examples with HIP back-end
````````````````````````````````````

As of now, the back-end has only been tested on the NVIDIA platform.

* NVIDIA Platform

  * One issue in this branch of alpaka is that the host compiler flags don't propagate to the device compiler, as they do in CUDA. This is because a counterpart to the ``CUDA_PROPAGATE_HOST_FLAGS`` cmake variable has not been defined in the FindHIP.cmake file.
    alpaka forwards the host compiler flags in cmake to the ``HIP_NVCC_FLAGS`` cmake variable, which also takes user-given flags. To add flags to this variable, toggle the advanced mode in ``ccmake``.


Random Number Generator Library rocRAND for HIP back-end
````````````````````````````````````````````````````````

*rocRAND* provides an interface for HIP, where the cuRAND or rocRAND API is called depending on the chosen HIP platform (can be configured with cmake in alpaka).

Clone the rocRAND repository, then build and install it

.. code-block::

  git clone https://github.com/ROCmSoftwarePlatform/rocRAND
  cd rocRAND
  mkdir -p build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=${HIP_PATH} -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF -DCMAKE_MODULE_PATH=${HIP_PATH}/cmake ..
  make


The ``CMAKE_MODULE_PATH`` is a cmake variable for locating module finding scripts like *FindHIP.cmake*.
The paths to the *rocRAND* library and include directories should be appended to the ``CMAKE_PREFIX_PATH`` variable.
