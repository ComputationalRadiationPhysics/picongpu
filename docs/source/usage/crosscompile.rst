.. _crosscompile-riscv:

Cross-compile for RISC-V
========================

This section contains information on how to cross-compile for RISC-V.

This section assumes you have a x86 system with a compiled gnu/clang compiler which can target RISC-V.
A detailed description on how to setup can be find under `RISC-V-Toolchain`_.

.. _RISC-V-Toolchain: https://riscv.epcc.ed.ac.uk/issues/toolchains+debugging/

You must compile all dependencies static only!
We observed problems with the CMake MPI detection therefore we disable the CMake MPI search with `-DMPI_CXX_WORKS=ON`` and provide
all required compiler and linker parameters via the environment variable ``CXXFLAGS``, ``CFLAGS`` and ``LDFLAGS``.

.. code-block:: bash

   # set MPI_ROOT to the root MPI directory
   export LDFLAGS="-L$MPI_ROOT/lib -lmpi -lompitrace -lopen-rte -lopen-pal -ldl -lpthread -lutil -lrt"
   # compile for 64bit RISC-V with double floating point support
   export CXXFLAGS="-I$MPI_ROOT/include -pthread -march=rv64gc -mabi=lp64d"
   export CFLAGS=$CXXFLAGS

To be able to cross compile you should point to the CMake toolchain file shipped together with PIConGPU.
Depending on your environment, please select the Clang or GNU toolchain.
The execution backend is provided in this example explicitly.
We recommend to write a :ref:`profile <install-profile>` for the system.
Depending on the toolchain the environment variable ``RISCV_GNU_INSTALL_ROOT`` or ``RISCV_CLANG_INSTALL_ROOT`` should be provided.

.. code-block:: bash

   export RISCV_GNU_INSTALL_ROOT="$(dirname $(which riscv64-unknown-linux-gnu-gcc))/.."
   export RISCV_CLANG_INSTALL_ROOT="$(dirname $(which clang))/.."

.. code-block:: bash

   # ``omp2b`` for OpenMP and ``serial`` for serial execution on one core
   pic-build -b omp2b -c"-DMPI_CXX_WORKS=ON  -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_PICONGPU_SRC>/cmake/riscv64-gnu.toolchain.cmake"

After PIConGPU is compiled you can interactively join to the RISC-V compute node and test if you compiled for the right target architecture ``./bin/picongpu --help``.
After this short test you should follow the typical workflow and start your simulation via :ref:`TBG <usage-tbg>`.