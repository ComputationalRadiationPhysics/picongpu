.. _usage-plugins-countParticles:

Count Particles
---------------

This plugin counts the total number of *macro particles associated with a species* and writes them to a file for specified time steps. 
It is used mainly for debugging purposes. 
Only in case of constant particle density, where each macro particle describes the same number of real particles (weighting), conclusions on the plasma density can be drawn.

.cfg file
^^^^^^^^^

The *CountParticles* plugin is always complied for all species.
By specifying the perodicity of the output using the comand line argument ``--e_macroParticlesCount.period`` (here for an electron species called ``e``) with picongpu, the plugin is enabled.
Setting ``--e_macroParticlesCount.period 100`` adds the number of all electron like macro particles to the file `ElectronsCount.dat` for every 100th time step of the simulation.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations.

Host
""""

negligible.

Output
^^^^^^

In the output file ``e_macroParticlesCount.dat``, there are three columns.
The first is the integer number of the time step.
The second is the number of macro particles as integer - useful for exact counts.
And the third is the number of macro particles in scintific floating point notation - provides better human readability.

Known Issues
^^^^^^^^^^^^

Currently, the file ``e_macroParticlesCount.dat``  is overwritten when restarting the simulation. 
Therefore, all previously stored counts are lost.

