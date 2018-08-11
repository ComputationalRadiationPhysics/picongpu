.. _usage-tbg:

TBG
===

.. sectionauthor:: Axel Huebl
.. moduleauthor:: René Widera

Our tool *template batch generator* (``tbg``) abstracts program runtime options from technical details of supercomputers.
On a desktop PC, one can just execute a command interactively and instantaneously.
Contrarily on a supercomputer, resources need to be shared between different users efficiently via *job scheduling*.
Scheduling on today's supercomputers is usually done via *batch systems* that define various queues of resources.

An unfortunate aspect about batch systems from a user's perspective is, that their usage varies a lot.
And naturally, different systems have different resources in queues that need to be described.

We abstract the description of queues, resource acquisition and job submission away from PIConGPU user input via *template files* (``.tpl``).
For example, the ``.cfg`` file defines how many *devices* shall be used for computation, but the ``.tpl`` file calculates how many *physical nodes* will be requested.
Also, the ``.tpl`` file takes care off how to spawn a process when scheduled, e.g. with ``mpiexec`` and which flags for networking details need to be passed.
After combining the *machine independent* (portable) ``.cfg`` file from user input with the *machine dependent* ``.tpl`` file, ``tbg`` can submit the requested job to the batch system.

Last but not least, one usually wants to store the input of a simulation with its output.
``tbg`` conveniently automates this task before submission.

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

Batch System Examples
^^^^^^^^^^^^^^^^^^^^^

.. sectionauthor:: Axel Huebl, Richard Pausch

Slurm
"""""

Slurm is a modern batch system, e.g. installed on the Taurus cluster at TU Dresden.

.. include:: ../install/profiles/taurus-tud/Slurm_Tutorial.rst
   :start-line: 3

PBS
"""

PBS (for *Portable Batch System*) is a widely distributed batch system that comes in several implementations (open, professional, etc.).
It is used, e.g. on Hypnos at HZDR.

.. include:: ../install/profiles/hypnos-hzdr/PBS_Tutorial.rst
   :start-line: 3
