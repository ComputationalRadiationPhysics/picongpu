.. _usage-basics:

.. seealso::

   You need to have an :ref:`environment loaded <install-profile>` (``source $HOME/picongpu.profile``) that provides all :ref:`PIConGPU dependencies <install-dependencies>` to complete this chapter.

Basics
======

.. sectionauthor:: Axel Huebl

Preparation
-----------

First, decide where to store input files, a good place might be ``$HOME`` (``~``) because it is usually backed up.
Second, decide where to store your output of simulations which needs to be placed on a high-bandwidth, large-storage file system which we will refer to as ``$SCRATCH``.

For a first test you can also use your home directory:

.. code-block:: bash

   export SCRATCH=$HOME

We need a few directories to structure our workflow:

.. code-block:: bash

    # PIConGPU input files
    mkdir $HOME/picInputs

    # PIConGPU simulation output
    mkdir $SCRATCH/runs

Step-by-Step
------------

1. Create an Input (Parameter) Set
""""""""""""""""""""""""""""""""""

.. code-block:: bash
   :emphasize-lines: 2

   # clone the LWFA example to $HOME/picInputs/myLWFA
   pic-create $PIC_EXAMPLES/LaserWakefield $HOME/picInputs/myLWFA

   # switch to your input directory
   cd $HOME/picInputs/myLWFA

PIConGPU is controlled via two kinds of input sets: compile-time options and runtime options.

Edit the :ref:`.param files <usage-params>` inside ``include/picongpu/param/``.
Initially and when options are changed, PIConGPU *requires a re-compile*.

Now edit the :ref:`runtime (command line) arguments <usage-cfg>` in ``etc/picongpu/*.cfg``.
These options do *not* require a re-compile when changed (e.g. simulation size, number of GPUs, plugins, ...).

2. Compile Simulation
"""""""""""""""""""""

In our input, ``.param`` files are build directly into the PIConGPU binary for performance reasons.
Changing or initially adding those requires a compile.

In this step you can optimize the simulation for the specific hardware you want to run on.
By default, we compile for Nvidia GPUs with the CUDA backend, targeting the oldest compatible `architecture <https://developer.nvidia.com/cuda-gpus>`_.

.. code-block:: bash
   :emphasize-lines: 1

   pic-build

This step will take a few minutes.
Time for a coffee or a `sword fight <https://xkcd.com/303/>`_!

3. Run Simulation
"""""""""""""""""

While you are still in ``$HOME/picInputs/myLWFA``, start your simulation on one CUDA capable GPU:

.. code-block:: bash
   :emphasize-lines: 2
   
   # example run for an interactive simulation on the same machine
   tbg -s bash -c etc/picongpu/0001gpus.cfg -t etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/lwfa_001

This will create the directory ``$SCRATCH/runs/lwfa_001`` where all simulation output will be written to.
``tbg`` will further create a subfolder ``input/`` in the directory of the run with the same structure as ``myLWFA`` to archive your input files.

Further Reading
---------------

Individual input files, their syntax and usage are explained in the following sections.

See ``tbg --help`` :ref:`for more information <usage-tbg>` about the ``tbg`` tool.

For example, if you want to run on the HPC System `"Hypnos" at HZDR <https://www.hzdr.de/db/Cms?pOid=12231>`_, your tbg submit command would just change to:

.. code-block:: bash
   :emphasize-lines: 2

   # request 16 GPUs from the PBS batch system and run on the queue k20
   tbg -s qsub -c etc/picongpu/0016gpus.cfg -t etc/picongpu/hypnos-hzdr/k20.tpl $SCRATCH/runs/lwfa_002

pic-create
""""""""""

This tool is just a short-hand to create a new set of input files.
It does a copy from an already existing set of input files (e.g. our examples or a previous simulation) and adds additional default files.

See ``pic-create --help`` for more options during input set creation:

.. program-output:: ../../pic-create --help

A run simulation can also be reused to create derived input sets via ``pic-create``:

.. code-block:: bash

   pic-create $SCRATCH/runs/lwfa_001/input $HOME/picInputs/mySecondLWFA

pic-build
"""""""""

This tool is actually a short-hand for an :ref:`out-of-source build with CMake <install-source>`.

In detail, it does:

.. code-block:: bash
   :emphasize-lines: 6,11

   # go to an empty build directory
   mkdir -p .build
   cd .build

   # configure with CMake
   pic-configure $OPTIONS ..

   # compile PIConGPU with the current input set (e.g. myLWFA)
   # - "make -j install" runs implicitly "make -j" and then "make install"
   # - make install copies resulting binaries to input set
   make -j install

``pic-build`` accepts the same command line flags as ``pic-configure``.
For example, if you want to build for running on CPUs instead of a GPUs, call:

.. code-block:: bash
   :emphasize-lines: 2

   # example for running efficiently on the CPU you are currently compiling on
   pic-build -b "omp2b"

Its full documentation from ``pic-build --help`` reads:

.. program-output:: ../../pic-build --help

pic-configure
"""""""""""""

The tools is just a convenient wrapper for a call to `CMake <https://cmake.org>`_.
It is executed from an empty build directory.
You will likely not use this tool directly when using ``pic-build`` from above.

We *strongly recommend* to set the appropriate target compute backend via ``-b`` for optimal performance.
For Nvidia CUDA GPUs, set the `compute capability <https://developer.nvidia.com/cuda-gpus>`_ of your GPU:

.. code-block:: bash

   # example for running efficiently on a K80 GPU with compute capability 3.7
   pic-configure -b "cuda:37" $HOME/picInputs/myLWFA

For running on a CPU instead of a GPU, set this:

.. code-block:: bash

   # example for running efficiently on the CPU you are currently compiling on
   pic-configure -b "omp2b:native" $HOME/picInputs/myLWFA

.. note::

   If you are compiling on a cluster, the CPU architecture of the head/login nodes versus the actual compute architecture does likely vary!
   Compiling a backend for the wrong architecture does in the best case dramatically reduce your performance and in the worst case will not run at all!

   During configure, the backend's architecture is forwarded to the compiler's ``-mtune`` and ``-march`` flags.
   For example, if you are `compiling with GCC <https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html>`_ for running on *AMD Opteron 6276 CPUs* set ``-b omp2b:bdver1`` or for *Intel Xeon Phi Knight's Landing CPUs* set ``-b omp2b:knl``.

See ``pic-configure --help`` for more options during input set configuration:

.. program-output:: ../../pic-configure --help

After running configure you can run ``ccmake .`` to set additional compile options (optimizations, debug levels, hardware version, etc.).
This will influence your build done via ``make``.

You can pass further options to configure PIConGPU directly instead of using ``ccmake .``, by passing ``-c "-DOPTION1=VALUE1 -DOPTION2=VALUE2"``.
