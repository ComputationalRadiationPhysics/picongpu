.. _usage-basics:

.. sectionauthor:: Axel Huebl

.. seealso::

   You need to have an :ref:`environment loaded <install-profile>` (``. $HOME/picongpu.profile``) that provides all :ref:`PIConGPU dependencies <install-dependencies>` to complete this chapter.

Basics
======

Preparation
-----------

First, decide where to store input files, a good place might be ``$HOME`` (``~``) because it is usually backed up.
Second, decide where to store your output of simulations which needs to be placed on a high-bandwidth, large-storage file system which we will refer to as ``$SCRATCH``.

As in our :ref:`compiling from source <install-source>` section, we need a few directories to structure our workflow:

.. code-block:: bash

    # source code
    mkdir $HOME/src
    # temporary build directory
    mkdir $HOME/build

    # PIConGPU input files
    mkdir $HOME/paramSets
    # PIConGPU simulation output
    mkdir $SCRATCH/runs


Step-by-Step
------------

TL;DR
"""""

("*too long, didn't read* and know how :ref:`compiling works <install-source>`")

.. code-block:: bash
   :emphasize-lines: 1,4,5,8

   $PICSRC/createParameterSet ~/paramSets/originalSet ~/paramSets/myLWFA
   
   cd ~/build
   $PICSRC/configure $HOME/paramSets/myLWFA
   make -j install
   
   cd ~/paramSets/myLWFA
   tbg -s qsub -c submit/0016gpus.cfg -t submit/hypnos-hzdr/k20_profile.tpl $SCRATCH/runs/lwfa_001


1. Create an Input (Parameter) Set
""""""""""""""""""""""""""""""""""

.. code-block:: bash
   :emphasize-lines: 2

   # clone the LWFA example to $HOME/paramSets/myLWFA
   $PICSRC/createParameterSet $PICSRC/examples/LaserWakefield/ $HOME/paramSets/myLWFA

Now edit ``$HOME/paramSets/case001/include/simulation_defines/param/*`` to change the :ref:`physical configuration of this parameter set <usage-params>`.

Now edit ``$HOME/paramSets/case001/submit/*.cfg`` to adjust :ref:`runtime parameters (simulation size, number of GPUs, plugins, ...) <usage-cfg>`.

Hint: you can further create parameter sets from parameter sets.

2. Compile Simulation
"""""""""""""""""""""

New ``.param`` files in inputs or changes of parameters in excisting files require a re-compile of PIConGPU.
Our script ``$PICSRC/configure`` is a wrapper for CMake to quickly specify which parameter set and source version of PIConGPU shall be used.

.. code-block:: bash
   :emphasize-lines: 7,12

   # go to an empty build directory
   cd $HOME/build
   # clean it if necessary
   rm -rf ../build/*

   # configure case001
   $PICSRC/configure $HOME/paramSets/myLWFA

   # compile PIConGPU with the current parameter set (myLWFA)
   # - "make -j install" runs implicitly "make -j" and then "make install"
   # - make install copies resulting binaries to parameter set
   make -j install

We always configure *one* parameter set for *one* compilation.
If you adjust ``.param`` input files just now, you can just go back to ``$HOME/build`` and run ``make -j install`` again without further need to clean the directory or configuration.

3. Run Simulation
"""""""""""""""""

.. code-block:: bash
   :emphasize-lines: 5

   # go to param set with up-to-date PIConGPU binaries
   cd $HOME/paramSets/myLWFA
   
   # example run for the HPC System "hypnos" using a PBS batch system
   tbg -s qsub -c submit/0016gpus.cfg -t submit/hypnos-hzdr/k20_profile.tpl $SCRATCH/runs/lwfa_001

This will create the directory ``$SCRATCH/runs/lwfa_001`` were all simulation output will be written to.
``tbg`` will further create a subfolder ``picongpu/`` in the directory of the run with the same structure as ``myLWFA`` to archive your input files.

Further Reading
---------------

Individual input files, their syntax and usage are explained in the following sections.

See ``$PICSRC/createParameterSet --help`` for more options during parameter set creation:

.. program-output:: ../../createParameterSet --help

See ``$PICSRC/configure --help`` for more options during parameter set configuration:

.. program-output:: ../../configure --help

After running configure you can run ``ccmake .`` to set additional compile options (optimizations, debug levels, hardware version, etc.).
This will influence your build done via ``make``.

You can pass further options to configure PIConGPU directly instead of using ``ccmake .``, by passing ``-c "-DOPTION1=VALUE1 -DOPTION2=VALUE2"``.

The ``picongpu/`` directory of a run can also be reused to clone parameters via ``createParameterSet`` by using this run as origin directory or to create a new binary with ``configure``: e.g. ``$PICSRC/configure -i $HOME/paramSets/myLWFA2 $SCRATCH/runs/lwfa_001``.

See ``tbg --help`` :ref:`for more information <usage-tbg>` about the ``tbg`` tool.
