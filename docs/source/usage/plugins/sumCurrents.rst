.. _usage-plugins-sumCurrents:

Sum Currents
------------

This plugin computes the total current integrated/added over the entire volume simulated.

.cfg file
^^^^^^^^^

The plugin can be activated by setting a non-zero value with the command line flag ``--sumcurr.period``.
The value set with ``--sumcurr.period`` is the periodicity, at which the total current is computed.
E.g. ``--sumcurr.period 100`` computes and prints the total current for time step *0, 100, 200, ...*.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

negligible.

Host
""""

negligible.

Output
^^^^^^

The result is printed to *standard output*. 
Therefore, it goes both to ``./simOutput/output`` as well as to the output file specified by the machine used (usually the ``stdout`` file in the main directory of the simulation).
The output is ASCII-text only.
It has the following format:

.. code::

   [ANALYSIS] [_rank] [COUNTER] [SumCurrents] [_currentTimeStep] {_current.x _current.y _current.z} Abs:_absCurrent

============================================ ============================================== =================
Value                                        Description                                    Unit
============================================ ============================================== =================
``_rank``                                    MPI rank at which prints the particle position *none*
``_currentTimeStep``                         simulation time step = number of PIC cycles    *none*
``_current.x`` ``_current.y`` ``_current.z`` electric current                               Ampere per second
``_absCurrent``                              magnitude of current                           Ampere per second
============================================ ============================================== =================

In order to extract only the total current information from the output stored in `stdout`, the following command on a bash command line could be used:

.. code:: bash

   grep SumCurrents stdout > totalCurrent.dat

The plugin data is then stored in ``totalCurrent.dat``.

Known Issues
^^^^^^^^^^^^

Currently, both ``output`` and ``stdout`` are overwritten at restart. 
All data from the plugin is lost, if these file are not backuped manually. 
