.. _usage-plugins-countPerSupercell:

Count per Supercell
-------------------

This plugin counts the total number of *macro particles of a species* for each super cell and sstores the result in an hdf5 file. 
Only in case of constant particle density, where each macro particle describes the same number of real particles (weighting), conclusions on the plasma density can be drawn.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.cfg files
^^^^^^^^^^

By specifying the perodicity of the output using the comand line argument ``--e_macroParticlesPerSuperCell.period`` (here for an electron species ``e``) with picongpu, the plugin is enabled.
Setting ``--e_macroParticlesPerSuperCell.period 100`` adds the number of all electron like macro particles to the file ``e_macroParticlesCount.dat`` for every 100th time step of the simulation.

Accelerator
"""""""""""

an extra permanent allocation of ``size_t`` for each local supercell.

Host
""""

negligible.

Output
^^^^^^

The output is stored as hdf5 file in a separate directory.
