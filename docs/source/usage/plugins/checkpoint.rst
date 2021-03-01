.. _usage-plugins-checkpoint:

Checkpoint
----------

Stores the primary data of the simulation for restarts.
Primary data includes:

* electro-magnetic fields
* particle attributes
* state of random number generators and particle ID generator
* ...

.. note::

   Some plugins have their own internal state.
   They will be notified on checkpoints to store their state themselves.

What is the format of the created files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We write our fields and particles in an open markup called :ref:`openPMD <pp-openPMD>`.

For further details, see the according sections in :ref:`the openPMD API <usage-plugins-openPMD>` and :ref:`ADIOS <usage-plugins-ADIOS>`.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`openPMD API or ADIOS libraries <install-dependencies>` are compiled in.

.cfg file
^^^^^^^^^

You can use ``--checkpoint.period`` to specify the output period of the created checkpoints.
Note that this plugin will only be available if the openPMD API, libSplash (HDF5) or ADIOS is found during compile configuration.

============================================= ======================================================================================
PIConGPU command line option                  Description
============================================= ======================================================================================
``--checkpoint.period <N>``                   Create checkpoints every N steps.
``--checkpoint.backend <IO-backend>``         IO-backend used to create the checkpoint.
``--checkpoint.directory <string>``           Directory inside ``simOutput`` for writing checkpoints.
                                              Default is ``checkpoints``.
``--checkpoint.file <string>``                Relative or absolute fileset prefix for writing checkpoints.
                                              If relative, checkpoint files are stored under ``simOutput/<checkpoint-directory>``.
                                              Default depends on the selected IO-backend.
``--checkpoint.restart``                      Restart a simulation from the latest checkpoint.
``--checkpoint.restart.step <N>``             Select a specific restart checkpoint.
``--checkpoint.restart.backend <IO-backend>`` IO-backend used to load a existent checkpoint.
``--checkpoint.restart.directory <string>``   Directory inside ``simOutput`` containing checkpoints for a restart.
                                              Default is ``checkpoints``.
``--checkpoint.restart.file <string>``        Relative or absolute fileset prefix for reading checkpoints.
                                              If relative, checkpoint files are searched under ``simOutput/<checkpoint-directory>``.
                                              Default depends on the selected IO-backend``.
``--checkpoint.restart.chunkSize <N>``        Number of particles processed in one kernel call during restart to prevent frame count
                                              blowup.
``--checkpoint.restart.loop <N>``             Number of times to restart the simulation after simulation has finished.
                                              This mode is intended for visualization and not all plugins support it.
``--checkpoint.<IO-backend>.*``               Additional options to control the IO-backend
============================================= ======================================================================================

Depending on the available external dependencies (see above), the options for the `<IO-backend>` are:

* :ref:`openPMD <usage-plugins-openPMD>`
* :ref:`adios <usage-plugins-ADIOS>` (keep in mind the :ref:`note on meta-files <usage-plugins-ADIOS-meta>` for restarts)

Interacting Manually with Checkpoint Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Interacting with the *raw data of checkpoints* for manual manipulation is considered an advanced feature for experienced users.

Contrary to regular output, checkpoints contain additional data which might be confusing on the first glance.
For example, some comments might be missing, all data from our concept of `slides for moving window simulations <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions>`_ will be visible, additional data for internal states of helper classes is stored as well and index tables such as openPMD particle patches are essential for parallel restarts.
