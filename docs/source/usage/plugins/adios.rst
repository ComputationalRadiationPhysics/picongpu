.. _usage-plugins-ADIOS:

ADIOS
-----

Stores simulation data such as fields and particles as `ADIOS <https://www.olcf.ornl.gov/center-projects/adios>`_ files or ADIOS staging methods.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`ADIOS library <install-dependencies>` is compiled in.

.param file
^^^^^^^^^^^

The corresponding ``.param`` file is :ref:`fileOutput.param <usage-params-plugins>`.

One can e.g. disable the output of particles by setting:

.. code-block:: cpp

   /* output all species */
   using FileOutputParticles = VectorAllSpecies;
   /* disable */
   using FileOutputParticles = bmpl::vector0< >;

.cfg file
^^^^^^^^^

You can use ``--adios.period`` and ``--adios.file`` to specify the output period and path and name of the created fileset.
For example, ``--adios.period 128 --adios.file simData`` will write the simulation data to files of the form ``simData_0.bp``, ``simData_128.bp`` in the default simulation output directory every 128 steps.
Note that this plugin will only be available if ADIOS is found during compile configuration.

============================ ==================================================================================================================================================================
PIConGPU command line option description
============================ ==================================================================================================================================================================
``--adios.period``           Period after which simulation data should be stored on disk. Default is ``0``, which means that no data is stored.
``--adios.file``             Relative or absolute fileset prefix for simulation data. If relative, files are stored under ``simOutput``. Default is ``simDataAdios``.
``--adios.compression``      Set data transform compression method. See ``adios_config -m`` for which compression methods are available. This flag also influences compression for checkpoints.
``--adios.aggregators``      Set number of I/O aggregator nodes for ADIOS ``MPI_AGGREGATE`` transport method.
``--adios.ost``              Set number of I/O OSTs for ADIOS ``MPI_AGGREGATE`` transport method.
============================ ==================================================================================================================================================================

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.
