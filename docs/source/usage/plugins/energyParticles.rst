.. _usage-plugins-energyParticles:

Energy Particles
----------------

This plugin computes the **kinetic and total energy summed over all particles** of a species for time steps specified. 

.cfg file
^^^^^^^^^

Only the time steps at which the total kinetic energy of all particles should be specified needs to be set via command line argument.

================================ ========================================================================================================
PIConGPU command line option     Description
================================ ========================================================================================================
``--e_energy.period 100``        Sets the time step period at which the energy of all **electrons** in the simulation should be simulated.
                                 If set to e.g. ``100``, the energy is computed for time steps *0, 100, 200, ...*.
                                 The default value is ``0``, meaning that the plugin does not compute the particle energy.
``--<species>_energy.period 42`` Same as above, for any other species available.
``--<species>_energy.filter``    Use filtered particles. All available filters will be shown with ``picongpu --help``
================================ ========================================================================================================

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

The plugin creates files prefixed with the species' name and the filter name as postfix, e.g. `e_energy_<filterName>.dat` for the electron energies and `p_energy_<filterName>.dat` for proton energies.
The file contains a header describing the columns.

.. code::

   #step Ekin_Joule E_Joule
   0.0   0.0        0.0

Following the header, each line is the output of one time step.
The time step is given as first value.
The second value is the kinetic energy of all particles at that time step. And the last value is the total energy (kinetic + rest energy) of all particles at that time step.

Example Visualization
^^^^^^^^^^^^^^^^^^^^^

Python snippet:

.. code::

   import numpy as np

   simDir = "path/to/simOutput/"

   # Ekin in Joules (see EnergyParticles)
   e_sum_ene = np.loadtxt(simDir + "e_energy_all.dat")[:, 0:2]
   p_sum_ene = np.loadtxt(simDir + "p_energy_all.dat")[:, 0:2]
   C_sum_ene = np.loadtxt(simDir + "C_energy_all.dat")[:, 0:2]
   N_sum_ene = np.loadtxt(simDir + "N_energy_all.dat")[:, 0:2]
   # Etotal in Joules
   fields_sum_ene = np.loadtxt(simDir + "fields_energy.dat")[:, 0:2]

   plt.figure()
   plt.plot(e_sum_ene[:,0], e_sum_ene[:,1], label="e")
   plt.plot(p_sum_ene[:,0], p_sum_ene[:,1], label="p")
   plt.plot(C_sum_ene[:,0], C_sum_ene[:,1], label="C")
   plt.plot(N_sum_ene[:,0], N_sum_ene[:,1], label="N")
   plt.plot(fields_sum_ene[:,0], fields_sum_ene[:,1], label="fields")
   plt.plot(
       e_sum_ene[:,0],
       e_sum_ene[:,1] + p_sum_ene[:,1] + C_sum_ene[:,1] + N_sum_ene[:,1] + fields_sum_ene[:,1],
       label="sum"
   )
   plt.legend()
