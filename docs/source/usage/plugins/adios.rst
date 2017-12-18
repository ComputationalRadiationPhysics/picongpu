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
For example, ``--adios.period 128 --adios.file simData --adios.source 'species_all'`` will write only the particle species data to files of the form ``simData_0.bp``, ``simData_128.bp`` in the default simulation output directory every 128 steps.
Note that this plugin will only be available if ADIOS is found during compile configuration.

============================ ==================================================================================================================================================================
PIConGPU command line option description
============================ ==================================================================================================================================================================
``--adios.period``           Period after which simulation data should be stored on disk.
``--adios.file``             Relative or absolute fileset prefix for simulation data. If relative, files are stored under ``simOutput``.
``--adios.compression``      Set data transform compression method. See ``adios_config -m`` for which compression methods are available. This flag also influences compression for checkpoints.
``--adios.aggregators``      Set number of I/O aggregator nodes for ADIOS ``MPI_AGGREGATE`` transport method.
``--adios.ost``              Set number of I/O OSTs for ADIOS ``MPI_AGGREGATE`` transport method.
``--adios.disable-meta``     Disable on-defly creation of the adios journal file. Allowed values ``0`` means write a journal file, ``1`` skips its generation.
``--adios.source``           Select data sources to dump. Default is ``species_all,fields_all``, which dumps all fields and particle species.
============================ ==================================================================================================================================================================

.. note::

   This plugin is a multi plugin. 
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   In the case where a optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   e.g.

   .. code-block:: bash

      --adios.period 128 --adios.file simData1 --adios.source 'species_all' 
      --adios.period 1000 --adios.file simData2 --adios.source 'fields_all' --adios.disable-meta 1

   creates two plugins:

   #. dump all species data each 128th time step, **do not create** the adios journal meta file.
   #. dump all field data each 1000th time step but **create** the adios journal meta file.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations.

Host
""""

as soon as ADIOS is compiled in, one extra ``mallocMC`` heap for the particle buffer is permanently reserved.
During I/O, particle attributes are allocated one after another.

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.
