.. _usage-tbg:

TBG
===

.. sectionauthor:: Axel Huebl, Richard Pausch
.. moduleauthor:: René Widera

todo: explain idea and use case

- what is a batch system
- cfg files
- tpl files
- behaviour (existing dirs, submission, environment)


Usage
^^^^^

.. program-output:: ../../src/tools/bin/tbg --help

.. _usage-cfg:

.cfg File Macros
^^^^^^^^^^^^^^^^

Feel free to copy & paste sections of the files below into your `.cfg`, e.g. to configure complex plugins:

.. literalinclude:: ../../TBG_macros.cfg
   :language: bash

Batch System Examples
^^^^^^^^^^^^^^^^^^^^^

Slurm
"""""

Slurm is a modern batch system, e.g. installed on the Taurus cluster at TU Dresden.

.. include:: ../install/submit/taurus-tud/Slurm_Tutorial.rst
   :start-line: 3

PBS
"""

PBS (for *Portable Batch System*) is a widely distributed batch system that comes in several implementations (open, professional, etc.).
It is used, e.g. on Hypnos at HZDR.

.. include:: ../install/submit/hypnos-hzdr/PBS_Tutorial.rst
   :start-line: 3
