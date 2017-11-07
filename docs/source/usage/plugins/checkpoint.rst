.. _usage-plugins-checkpoint:

Checkpoint
----

Stores the primary data of the simulation for restarts.
Primary data includes:

* electro-magnetic fields
* particle attributes
* state of random number generators and particle ID generator
* ...

.. note::

   Some plugins have their onw internal state.
   They will be notified on checkpoints to store their state themselves.

What is the format of the created files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We write our fields and particles in an open markup called **openPMD**.
You can investigate your files via a large collection of `tools and frameworks <https://github.com/openPMD/openPMD-projects>`_ or your use the native HDF5 bindings of your `favourite programming language <https://en.wikipedia.org/wiki/Hierarchical_Data_Format#Interfaces>`_.

**Resources for a quick-start:**

* `online tutorial <http://www.openPMD.org>`_
* `example files <https://github.com/openPMD/openPMD-example-datasets>`_
* `written standard <https://github.com/openPMD/openPMD-standard>`_ of the openPMD standard
* `list of projects <https://github.com/openPMD/openPMD-projects>`_ supporting openPMD files

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.cfg file
^^^^^^^^^

You can use ``--checkpoint.period`` to specify the output period of the created checkpoints.
Note that this plugin will only be available if ``libSplash and HDF5`` or Adios is found during compile configuration.

================================== ======================================================================================
PIConGPU command line option       Description
================================== ======================================================================================
``--checkpoint.backend``           IO-backend used to create the checkpoint.
``--checkpoint.file``              Relative or absolute fileset prefix for writing checkpoints.
                                   If relative, checkpoint files are stored under ``simOutput/<checkpoint-directory>``.
                                   Default depends on the selected IO-backend.
``--checkpoint.restart.backend``   IO-backend used to load a existent checkpoint.
``--checkpoint.restart.file``      Relative or absolute fileset prefix for reading checkpoints.
                                   If relative, checkpoint files are searched under ``simOutput/<checkpoint-directory>``.
                                   Default depends on the selected IO-backend``.
``--checkpoint.restart.chunkSize`` Number of particles processed in one kernel call during restart to prevent frame count blowup.
``--checkpoint.<IO-backend>.*      Additional options to control the IO-bakend
================================== ======================================================================================

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.
