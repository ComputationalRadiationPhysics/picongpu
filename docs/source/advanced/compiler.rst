Compiler Specifics
==================

Alpaka supports a large number of different compilers. Each of the compilers has its own special features. This page explains some of these specifics.

Choosing the correct Standard Library in Clang
++++++++++++++++++++++++++++++++++++++++++++++

Clang supports both, ``libstdc++`` shipped with the GNU GCC toolchain and ``libc++``, LLVM's own implementation of the C++ standard library. By default, clang and all clang-based compilers, such as the ``hipcc``, use ``libstdc++`` (GNU GCC). If more than one ``GCC`` version is installed, it is not entirely clear which version of ``libstdc++`` is selected. The following code can be used to check which standard library and version clang is using by default with the current setup.

.. code-block:: c++
    
    #include <iostream>

    int main(){
        #ifdef _GLIBCXX_RELEASE
        std::cout << "use libstdc++ (GNU GCC's standard library implementation)" << std::endl;
        std::cout << "version: " <<   _GLIBCXX_RELEASE << std::endl;
        #endif

        #ifdef _LIBCPP_VERSION
        std::cout << "use libc++ (LLVM's standard library implementation)" << std::endl;
        std::cout << "version: " <<   _LIBCPP_VERSION << std::endl;
        #endif
    }

The command ``clang -v ...`` shows the include paths and also gives information about the standard library used.

Choose a specific libstdc++ version
-----------------------------------

Clang provides the argument ``--gcc-toolchain=<path>`` which allows you to select the path of a GCC installation. For example, if you built the ``GCC`` compiler from source, you can select the installation prefix, which is the base folder with the subfolders ``include``, ``lib`` and so on.

If you are using CMake, you can set the ``--gcc-toolchain`` flag via the following CMake command line argument:

* ``-DCMAKE_CXX_FLAGS="--gcc-toolchain=<path>"`` if you use Clang as compiler for CPU backends or the HIP backend.
* ``-DCMAKE_CUDA_FLAGS="--gcc-toolchain=<path>"`` if you use Clang as CUDA compiler.

.. hint:: If you are using Ubuntu and install a new gcc version via apt, it is not possible to select a specific gcc version because apt installs all headers and shared libraries in subfolders of ``/usr/include`` and ``/usr/lib``. Therefore, you can only use the ``/usr`` base path and Clang will automatically select one of the installed libstdc++ versions.

.. hint:: If you installed Clang/LLVM with spack and a gcc compiler, the Clang compiler will use the ``libstdc++`` of the compiler used to build Clang/LLVM.


Selecting libc++
----------------

``libc++`` can be used if you set the compiler flag ``-stdlib=libc++``.

If you are using CMake, you can select ``libc++`` via the following CMake command line argument:

* ``-DCMAKE_CXX_FLAGS="-stdlib=libc++"`` if you use Clang as compiler for CPU backends or the HIP backend.
* ``-DCMAKE_CUDA_FLAGS="-stdlib=libc++"`` if you use Clang as CUDA compiler.
