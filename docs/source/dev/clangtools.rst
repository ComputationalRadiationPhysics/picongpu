.. _development-clangtools:

Clang Tools
===========

.. sectionauthor:: Axel Huebl

We are currently integrating support for Clang Tools [ClangTools]_ such as ``clang-tidy`` and ``clang-format``.
Clang Tools are fantastic for static source code analysis, e.g. to find defects, automate style formatting or modernize code.

Install
-------

At least LLVM/Clang 3.9 or newer is required.
On Debian/Ubuntu, install them via:


.. code-block:: bash

   sudo apt-get install clang-tidy-3.9

Usage
-----

Currently, those tools work only with CPU backends of PIConGPU.
For example, enable the *OpenMP* backend via:

.. code-block:: bash

   # in an example
   mkdir .build
   cd build

   pic-configure -c"-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON" ..

We try to auto-detect ``clang-tidy``.
If that fails, you can set a manual hint to an adequate version via ``-DCLANG_TIDY_BIN`` in CMake:

.. code-block:: bash

   pic-configure -c"-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON -DCLANG_TIDY_BIN=$(which clang-tidy-3.9)" ..

If a proper version of ``clang-tidy`` is found, we add a new ``clang-tidy`` build target:

.. code-block:: bash

   # enable verbose output to see all warnings and errors
   make VERBOSE=true clang-tidy


.. [ClangTools]
        Online (2017), https://clang.llvm.org/docs/ClangTools.html

