.. highlight:: bash

Installation
============

**Installing dependencies**

alpaka requires **Boost** and a modern C++ compiler (g++, clang++, Visual C++, …). In order to install **Boost**:

On Linux:

.. code-block::

  # RPM
  sudo dnf install boost-devel
  # DEB
  sudo apt install libboost-all-dev

On macOS:

.. code-block::

  # Using Homebrew, https://brew.sh
  brew install boost
  # Using MacPorts, https://macports.org
  sudo port install boost

On Windows:

.. code-block::

  # Using vcpkg, https://github.com/microsoft/vcpkg
  vcpkg install boost

**CMake** is the preferred system for configuration the build tree, building and installing. In order to install **CMake**:

On Linux:

.. code-block::

  # RPM
  sudo dnf install cmake
  # DEB
  sudo apt install cmake

On macOS or Windows:

Download the installer from https://cmake.org/download/

**Dependencies to use specific backends**: Depending on your target platform you may need additional packages to compile and run alpaka.

- NVIDIA GPUs: CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
- AMD GPUs: ROCm / HIP (https://rocmdocs.amd.com/en/latest/index.html)
- Intel GPUs: OneAPI Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.9x3lnh)

Tests and Examples
++++++++++++++++++

The examples and tests can be compiled without installing alpaka. They will use alpaka headers from the source directory.

**Build and run examples:**

.. code-block::

  # ..
  cmake -Dalpaka_BUILD_EXAMPLES=ON ..
  cmake --build . -t vectorAdd
  ./example/vectorAdd/vectorAdd # execution

**Build and run tests:**

.. code-block::

  # ..
  cmake -DBUILD_TESTING=ON ..
  cmake --build .
  ctest

**Enable accelerators:**

Alpaka uses different accelerators to execute kernels on different processors. To use a specific accelerator in alpaka, two steps are required.

1. Enable the accelerator during the CMake configuration time of the project.
2. Select a specific accelerator in the source code.

By default, no accelerator is enabled because some combinations of compilers and accelerators do not work, see the table of `supported compilers <https://github.com/alpaka-group/alpaka#supported-compilers>`_. To enable an accelerator, you must set a CMake flag via ``cmake .. -Dalpaka_ACC_<acc>_ENABLE=ON`` when you create a new build. The following example shows how to enable the CUDA accelerator and build an alpaka project:

.. code-block::

  cmake -Dalpaka_ACC_GPU_CUDA_ENABLE=ON ...

In the overview of :doc:`cmake arguments </advanced/cmake>` you will find all CMake flags for activating the different accelerators. How to select an accelerator in the source code is described on the :doc:`example page </basic/example>`.

.. warning::

  If an accelerator is selected in the source code that is not activated during CMake configuration time, a compiler error occurs.


.. hint::

  When the test or examples are activated, the alpaka build system automatically activates the ``serial backend``, as it is needed for many tests. Therefore, the tests are run with the ``serial backend`` by default. If you want to test another backend, you have to activate it at CMake configuration time, for example the ``HIP`` backend: ``cmake .. -DBUILD_TESTING=ON -Dalpaka_ACC_GPU_HIP_ENABLE=ON``. Some alpaka tests use a selector algorithm to choose a specific accelerator for the test cases. The selector works with accelerator priorities. Therefore, it is recommended to enable only one accelerator for a build to make sure that the right one is used.


**Installing alpaka**

If user is going to create her/his own project/example outside the source tree alpaka should be installed. Since alpaka is a header only library compilation is not needed before installation.

.. code-block::

  # Clone alpaka from github.com
  git clone --branch 1.1.0 https://github.com/alpaka-group/alpaka.git
  cd alpaka
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/install/ ..
  cmake --install .
