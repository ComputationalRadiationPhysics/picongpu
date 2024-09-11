.. _usage-tbg:

TBG
===

.. sectionauthor:: Axel Huebl, Klaus Steiniger
.. moduleauthor:: René Widera

Our tool *template batch generator* (``tbg``) abstracts program runtime options from technical details of supercomputers.
On a desktop PC, one can just execute a command interactively and instantaneously.
Contrarily on a supercomputer, resources need to be shared between different users efficiently via *job scheduling*.
Scheduling on today's supercomputers is usually done via *batch systems* that define various queues of resources.

An unfortunate aspect about batch systems from a user's perspective is, that their usage varies a lot.
And naturally, different systems have different resources in queues that need to be described.

PIConGPU runtime options are described in *configuration files* (``.cfg``).
We abstract the description of queues, resource acquisition and job submission via *template files* (``.tpl``).
For example, a ``.cfg`` file defines how many *devices* shall be used for computation, but a ``.tpl`` file calculates how many *physical nodes* will be requested.
Also, ``.tpl`` files takes care of how to spawn a process when scheduled, e.g. with ``mpiexec`` and which flags for networking details need to be passed.
After combining the *machine independent* (portable) ``.cfg`` file from user input with the *machine dependent* ``.tpl`` file, ``tbg`` can submit the requested job to the batch system.

Last but not least, one usually wants to store the input of a simulation with its output.
``tbg`` conveniently automates this task before submission.
The ``.tpl`` and the ``.cfg`` files that were used to start the simulation can be found in ``<tbg destination dir>/tbg/`` and can be used together with the ``.param`` files from ``<tbg destination dir>/input/.../param/`` to recreate the simulation setup.

In summary, PIConGPU runtime options in ``.cfg`` files are portable to any machine.
When accessing a machine for the first time, one needs to write template ``.tpl`` files, abstractly describing how to run PIConGPU on the specific queue(s) of the batch system.
We ship such template files already for a set of supercomputers, interactive execution and many common batch systems.
See ``$PICSRC/etc/picongpu/`` and :ref:`our list of systems with .profile files <install-profile>` for details.


Usage
^^^^^

.. program-output:: ../../bin/tbg --help

.. _usage-cfg:

.cfg File Macros
^^^^^^^^^^^^^^^^

Feel free to copy & paste sections of the files below into your ``.cfg``, e.g. to configure complex plugins:

.. literalinclude:: ../../TBG_macros.cfg
   :language: bash


Automatic Grid Adjustment
^^^^^^^^^^^^^^^^^^^^^^^^^

It is important to note that PIConGPU has a feature to automatically adjust the grid to comply with PIConGPU's constraints.
This feature is ACTIVE by default, meaning that unless explicit disabled PIConGPU does not guarantee that the grid you submitted is actually run.
This can be deactivated by the commandline flag `--autoAdjustGrid 0`.
If automatic adjustment kicks in, it prints a warning.
If automatic adjustment is deactivated, the run will fail during the initialisation for incompatible parameters.

The checks performed on the given grid parameters are the following:

.. literalinclude:: ../../../include/picongpu/simulation/control/DomainAdjuster.hpp
   :language: C++
   :start-after: doc-include-start: automatic-grid-adjustment
   :end-before: doc-include-end: automatic-grid-adjustment
   :dedent:


Batch System Examples
^^^^^^^^^^^^^^^^^^^^^

.. sectionauthor:: Axel Huebl, Richard Pausch, Klaus Steiniger


Linux workstation
"""""""""""""""""

PIConGPU can run on your laptop or workstation, even if there is no dedicated GPU available.
In this case it will run on the CPU.

In order to run PIConGPU on your machine, use ``bash`` as the submit command, i.e.
``tbg -s bash -t etc/picongpu/bash/mpirun.tpl -c etc/picongpu/1.cfg $SCRATCH/picRuns/001``

Slurm
"""""

Slurm is a modern batch system, e.g. installed on the Taurus cluster at TU Dresden, Hemera at HZDR, Cori at NERSC, among others.

.. include:: ../install/profiles/taurus-tud/Slurm_Tutorial.rst
   :start-line: 3

LSF
"""

LSF (for *Load Sharing Facility*) is an IBM batch system (``bsub``/BSUB).
It is used, e.g. on Summit at ORNL.

.. include:: ../install/profiles/summit-ornl/LSF_Tutorial.rst
   :start-line: 3
