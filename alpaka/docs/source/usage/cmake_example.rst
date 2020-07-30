.. highlight:: bash

CMake Example
=============

You can integrate alpaka into your project via ``find_package()`` in your ``CMakeLists.txt``.
This requires, that you :doc:`install </install/instructions>` alpaka.
If you do not install alpaka in a default path such as ``/usr/local/`` you have to set the ``CMake`` argument ``-Dalpaka_ROOT=/path/to/alpaka/install``.

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.15)

   set(_TARGET_NAME helloWorld)
   project(${_TARGET_NAME})

   find_package(alpaka REQUIRED)

   alpaka_add_executable(${_TARGET_NAME} helloWorld.cpp)
   target_link_libraries(
     ${_TARGET_NAME}
     PUBLIC alpaka::alpaka)

.. literalinclude:: ../../../example/helloWorld/src/helloWorld.cpp
   :language: C++
   :caption: helloWorld.cpp
