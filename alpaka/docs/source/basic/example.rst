Code Example
============

The following example shows a small hello word example written with alpaka that can be run on different processors.

.. literalinclude:: ../../../example/helloWorld/src/helloWorld.cpp
   :language: C++
   :caption: helloWorld.cpp

Use alpaka in your project
++++++++++++++++++++++++++

We recommend to use CMake for integrating alpaka into your own project. There are two possible methods.

Use alpaka via ``find_package``
-------------------------------

The ``find_package`` method requires alpaka to be :doc:`installed </basic/install>` in a location where CMake can find it. 

.. hint::

    If you do not install alpaka in a default path such as ``/usr/local/`` you have to set the CMake argument ``-Dalpaka_ROOT=/path/to/alpaka/install``.

The following example shows a minimal example of a ``CMakeLists.txt`` that uses alpaka:

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.22)
   project("myexample" CXX)

   find_package(alpaka REQUIRED)
   alpaka_add_executable(${PROJECT_NAME} helloWorld.cpp)
   target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)

In the CMake configuration phase of the project, you must activate the accelerator you want to use:

.. code-block:: bash

    cd <path/to/the/project/root>
    mkdir build && cd build
    cmake .. -Dalpaka_ACC_GPU_CUDA_ENABLE=ON
    cmake --build .
    ./myexample

A complete list of CMake flags for the  accelerator can be found :doc:`here </advanced/cmake>`.

If the configuration was successful and CMake found the CUDA SDK, the C++ template accelerator type ``alpaka::AccGpuCudaRt`` is available.

Use alpaka via ``add_subdirectory``
-----------------------------------

The ``add_subdirectory`` method does not require alpaka to be installed. Instead, the alpaka project folder must be part of your project hierarchy. The following example expects alpaka to be found in the ``project_path/thirdParty/alpaka``:

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.22)
   project("myexample" CXX)

   add_subdirectory(thirdParty/alpaka)
   alpaka_add_executable(${PROJECT_NAME} helloWorld.cpp)
   target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)

The CMake configure and build commands are the same as for the ``find_package`` approach.
