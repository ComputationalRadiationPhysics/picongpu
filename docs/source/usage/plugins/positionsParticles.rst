.. _usage-plugins-positionsParticles:

Positions Particles
-------------------

This plugin prints out the *position, momentum, mass, macro particle weighting, electric charge and relativistic gamma factor* of a particle to ``stdout`` (usually inside the ``simOutput/output`` file).
**It only works with test simulations that have only one particle.**

.cfg file
^^^^^^^^^

By setting the command line flag ``--<species>_position.period`` to a non-zero number, the analyzer is used.
In order to get the particle trajectory for each time step the period needs to be set to ``1``, meaning e.g. ``--e_position.period 1`` for electrons.
If less output is needed, e.g. only every 10th time step, the period can be set to different values, e.g. ``--e_position.period 10``.

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

The electron trajectory is written directly to the *standard out*.
Therefore, it goes both to ``./simOutput/output`` as well as to the output file specified by the machine used (usually the ``stdout`` file in the main directory of the simulation).
The output is ASCII-text only.
It has the following format:

.. code::

   [ANALYSIS] [MPI_Rank] [COUNTER] [<species>_position] [currentTimeStep] currentTime {position.x position.y position.z} {momentum.x momentum.y momentum.z} mass weighting charge gamma

============================================== ===================================================== ======
Value                                          Description                                           Unit
============================================== ===================================================== ======
``MPI_Rank``                                   MPI rank at which prints the particle position        *none*
``COUNTER``                                    name of the plugin | always ``<species>_position``
``currentTimeStep``                            simulation time step = number of PIC cycles           *none*
``currentTime``                                simulation time in SI units                           seconds
``position.x`` ``_position.y`` ``_position.z`` location of the particle in space                     meters
``momentum.x`` ``_momentum.y`` ``_momentum.z`` momentum of particle                                  kg m/s
``mass``                                       mass of macro particle                                kg
``weighting``                                  number of electrons represented by the macro particle *none*
``charge``                                     charge of macro particle                              Coulomb
``gamma``                                      relativistic gamma factor of particle                 *none*
============================================== ===================================================== ======

.. code::

   # an example output line:
   [ANALYSIS] [2] [COUNTER] [e_position] [878] 1.46440742e-14 {1.032e-05 4.570851689815522e-05 5.2e-06} {0 -1.
   337873603181226e-21 0} 9.109382e-31 1 -1.602176e-19 4.999998569488525

In order to extract only the trajectory information from the total output stored in `stdout`, the following command on a bash command line could be used:

.. code:: bash

   grep "e_position" stdout > trajectory.dat

The particle data is then stored in ``trajectory.dat``.

In order to extract e.g. the position from this line the following can be used:

.. code:: bash

   cat trajectory.dat | awk '{print $7}' | sed -e "s/{//g" | sed -e 's/}//g' | sed -e 's/,/\t/g' > position.dat

Known Issues
^^^^^^^^^^^^

.. attention::

   This plugin only works correctly if a single particle is simulated.
   If more than one particle is simulated, the output becomes random, because only the information of one particle is printed.
   This plugin might be upgraded to work with multiple particles, but better use our HDF5 or ADIOS plugin instead and assign `particleId`s to individual particles.

.. attention::

   Currently, both `output` and `stdout`are overwritten at restart. 
   All data from the plugin is lost, if these file are not backuped manually.
