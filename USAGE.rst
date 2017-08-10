.. _usage-basics:

.. seealso::

   You need to have an :ref:`environment loaded <install-profile>` (``. $HOME/picongpu.profile``) that provides all :ref:`PIConGPU dependencies <install-dependencies>` to complete this chapter.

Basics
======

.. sectionauthor:: Axel Huebl

Preparation
-----------

First, decide where to store input files, a good place might be ``$HOME`` (``~``) because it is usually backed up.
Second, decide where to store your output of simulations which needs to be placed on a high-bandwidth, large-storage file system which we will refer to as ``$SCRATCH``.

For a first test you can also use your home directory ``$HOME`` as simulation directory ``$SCRATCH``.

As in our :ref:`compiling from source <install-source>` section, we need a few directories to structure our workflow:

.. code-block:: bash

    # temporary build directory
    mkdir $HOME/build

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
   pic-create $PICSRC/examples/LaserWakefield/ $HOME/picInputs/myLWFA

Now edit ``$HOME/picInputs/myLWFA/include/picongpu/simulation_defines/param/*`` to change the :ref:`physical configuration of this input set <usage-params>`.

Now edit ``$HOME/picInputs/myLWFA/etc/picongpu/*.cfg`` to adjust :ref:`runtime parameters (simulation size, number of GPUs, plugins, ...) <usage-cfg>`.

2. Compile Simulation
"""""""""""""""""""""

In our input, ``.param`` files are build directly into the PIConGPU binary.
Therefore, changing or initially adding those requires a compile.

In this step you can optimize the simulation for the specific hardware you want to run on.
By default, we compile for Nvidia GPUs with CUDA targeting the oldest compatible `architecture <https://developer.nvidia.com/cuda-gpus>`_.

.. code-block:: bash
   :emphasize-lines: 7,12

   # go to an empty build directory
   cd $HOME/build
   # clean it if necessary
   rm -rf ../build/*

   # configure case001
   pic-configure $HOME/picInputs/myLWFA

   # compile PIConGPU with the current input set (myLWFA)
   # - "make -j install" runs implicitly "make -j" and then "make install"
   # - make install copies resulting binaries to input set
   make -j install

We always configure *one* input set for *one* compilation.
If you adjust ``.param`` input files just now, you can just go back to ``$HOME/build`` and run ``make -j install`` again without further need to clean the directory or configuration.

3. Run Simulation
"""""""""""""""""

.. code-block:: bash
   :emphasize-lines: 5

   # go to param set with up-to-date PIConGPU binaries
   cd $HOME/picInputs/myLWFA
   
   # example run for the HPC System "hypnos" using a PBS batch system
   tbg -s qsub -c etc/picongpu/0016gpus.cfg -t etc/picongpu/hypnos-hzdr/k20_profile.tpl $SCRATCH/runs/lwfa_001

This will create the directory ``$SCRATCH/runs/lwfa_001`` were all simulation output will be written to.
``tbg`` will further create a subfolder ``input/`` in the directory of the run with the same structure as ``myLWFA`` to archive your input files.

Further Reading
---------------

Individual input files, their syntax and usage are explained in the following sections.

See ``tbg --help`` :ref:`for more information <usage-tbg>` about the ``tbg`` tool.

pic-create
""""""""""

This tool is just a short-hand to create a new set of input files.
It does a copy from an already existing set of input files (e.g. our examples or a previous simulation) and adds additional default files.

See ``pic-create --help`` for more options during input set creation:

.. program-output:: ../../pic-create --help

A run simulation can also be reused to create derived input sets via ``pic-create``:

.. code-block:: bash

   pic-create $SCRATCH/runs/lwfa_001/input $HOME/picInputs/mySecondLWFA

pic-configure
"""""""""""""

The tools is just a convenient wrapper for a call to `CMake <https://cmake.org>`_.

We *strongly recommend* to set the appropriate target compute architecture via ``-a`` for optimal performance.
For Nvidia CUDA GPUs, set the `compute capability <https://developer.nvidia.com/cuda-gpus>`_ of your GPU:

.. code-block:: bash

   # example for running efficiently on a K80 GPU with compute capability 3.7
   pic-configure -a "cuda:37" $HOME/picInputs/myLWFA

For running on a CPU instead of a GPU, set this:

.. code-block:: bash

   # example for running efficiently on the CPU you are currently compiling on
   pic-configure -a "omp2b:native" $HOME/picInputs/myLWFA

.. note::

   If you are compiling on a cluster, the CPU architecture of the head/login nodes versus the actual compute architecture does likely vary!
   Compiling for the wrong architecture does in the best case dramatically reduce your performance and in the worst case will not run at all!

   During configure, the architecture is forwarded to the compiler's ``-mtune`` and ``-march`` flags.
   For example, if you are compiling for running on AMD Opteron 6276 CPUs set ``-a omp2b:bdver1``.

See ``pic-configure --help`` for more options during input set configuration:

.. program-output:: ../../pic-configure --help

After running configure you can run ``ccmake .`` to set additional compile options (optimizations, debug levels, hardware version, etc.).
This will influence your build done via ``make``.

You can pass further options to configure PIConGPU directly instead of using ``ccmake .``, by passing ``-c "-DOPTION1=VALUE1 -DOPTION2=VALUE2"``.
