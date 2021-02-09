.. _usage-basics:

.. seealso::

   You need to have an :ref:`environment loaded <install-profile>` (``source $HOME/picongpu.profile`` when installing from source or ``spack load picongpu`` when using spack) that provides all :ref:`PIConGPU dependencies <install-dependencies>` to complete this chapter.

.. warning::

   PIConGPU source code is portable and can be compiled on all major operating systems.
   However, helper tools like ``pic-create`` and ``pic-build`` described in this section rely on Linux utilities and thus are not expected to work on other platforms out-of-the-box.
   Note that building and using PIConGPU on other operating systems is still possible but has to be done manually or with custom tools.
   This case is not covered in the documentation, but we can assist users with it when needed.

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

PIConGPU is controlled via two kinds of textual input sets: compile-time options and runtime options.

Compile-time :ref:`.param files <usage-params>` reside in ``include/picongpu/param/`` and define the physics case and deployed numerics.
After creation and whenever options are changed, PIConGPU *requires a re-compile*.
Feel free to take a look now, but we will later come back on how to :ref:`edit those files <usage-params-edit>`.

:ref:`Runtime (command line) arguments <usage-cfg>` are set in ``etc/picongpu/*.cfg`` files.
These options do *not* require a re-compile when changed (e.g. simulation size, number of devices, plugins, ...).

2. Compile Simulation
"""""""""""""""""""""

In our input, ``.param`` files are build directly into the PIConGPU binary for :ref:`performance reasons <usage-params-rationale>`.
A compile is required after changing or initially adding those files.

In this step you can optimize the simulation for the specific hardware you want to run on.
By default, we compile for Nvidia GPUs with the CUDA backend, targeting the oldest compatible `architecture <https://developer.nvidia.com/cuda-gpus>`_.

.. code-block:: bash
   :emphasize-lines: 1

   pic-build

This step will take a few minutes.
Time for a coffee or a `sword fight <https://xkcd.com/303/>`_!

We explain in the :ref:`details section <usage-basics-build>` below how to set further options, e.g. CPU targets or tuning for newer GPU architectures.

3. Run Simulation
"""""""""""""""""

While you are still in ``$HOME/picInputs/myLWFA``, start your simulation on one CUDA capable GPU:

.. code-block:: bash
   :emphasize-lines: 2

   # example run for an interactive simulation on the same machine
   tbg -s bash -c etc/picongpu/1.cfg -t etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/lwfa_001

This will create the directory ``$SCRATCH/runs/lwfa_001`` where all simulation output will be written to.
``tbg`` will further create a subfolder ``input/`` in the directory of the run with the same structure as ``myLWFA`` to archive your input files.
Subfolder ``simOutput/`` has all the simulation results.
Particularly, the simulation progress log is in ``simOutput/output``.

Details on the Commands Above
-----------------------------

.. _usage-basics-tbg:

tbg
"""

The ``tbg`` tool is explained in detail :ref:`in its own section <usage-tbg>`.
Its primary purpose is to abstract the options in runtime ``.cfg`` files from the technical details on how to run on various supercomputers.

For example, if you want to run on the HPC System `"Hemera" at HZDR <https://www.hzdr.de/db/Cms?pOid=12231>`_, your ``tbg`` submit command would just change to:

.. code-block:: bash
   :emphasize-lines: 2

   # request 1 GPU from the PBS batch system and run on the queue "k20"
   tbg -s sbatch -c etc/picongpu/1.cfg -t etc/picongpu/hemera-hzdr/k20.tpl $SCRATCH/runs/lwfa_002

   # run again, this time on 16 GPUs
   tbg -s sbatch -c etc/picongpu/16.cfg -t etc/picongpu/hemera-hzdr/k20.tpl $SCRATCH/runs/lwfa_003

Note that we can use the same ``1.cfg`` file, your input set is *portable*.

.. _usage-basics-create:

pic-create
""""""""""

This tool is just a short-hand to create a new set of input files.
It copies from an already existing set of input files (e.g. our examples or a previous simulation) and adds additional helper files.

See ``pic-create --help`` for more options during input set creation:

.. program-output:: ../../bin/pic-create --help

A run simulation can also be reused to create derived input sets via ``pic-create``:

.. code-block:: bash

   pic-create $SCRATCH/runs/lwfa_001/input $HOME/picInputs/mySecondLWFA

.. _usage-basics-build:

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

``pic-build`` accepts the same command line flags as :ref:`pic-configure <usage-basics-configure>`.
For example, if you want to build for running on CPUs instead of a GPUs, call:

.. code-block:: bash
   :emphasize-lines: 2

   # example for running efficiently on the CPU you are currently compiling on
   pic-build -b "omp2b"

Its full documentation from ``pic-build --help`` reads:

.. program-output:: ../../bin/pic-build --help

.. _usage-basics-configure:

pic-configure
"""""""""""""

This tool is just a convenient wrapper for a call to `CMake <https://cmake.org>`_.
It is executed from an :ref:`empty build directory <install-source>`.

You will likely not use this tool directly.
Instead, :ref:`pic-build <usage-basics-build>` from above calls ``pic-configure`` for you, forwarding its arguments.

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

.. program-output:: ../../bin/pic-configure --help

After running configure you can run ``ccmake .`` to set additional compile options (optimizations, debug levels, hardware version, etc.).
This will influence your build done via ``make install``.

You can pass further options to configure PIConGPU directly instead of using ``ccmake .``, by passing ``-c "-DOPTION1=VALUE1 -DOPTION2=VALUE2"``.
