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
   using FileOutputParticles = MakeSeq_t< >;

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
``--adios.transport-params`` Further options for transports, see ADIOS manual chapter 6.1.5. Lustre example: ``random_offset=1;stripe_count=4`` (FS chooses OST; user chooses striping factor).
``--adios.disable-meta``     Disable on-the-fly creation of the adios journal file. Allowed values: ``0`` means write a journal file, ``1`` skips its generation.
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

Compression
^^^^^^^^^^^

ADIOS supports various on-the-fly compression methods.
Typical options:

.. code-block:: bash

   # single-threaded, slow zlib
   --adios.compression zlib

   # 6x multi-threaded, fast zstd via blosc, bitshuffle pre-conditioner and compression threshold of 2kB
   --adios.compression blosc:threshold=2048,shuffle=bit,lvl=1,threads=6,compressor=zstd

See the `ADIOS manual <https://users.nccs.gov/~pnorbert/ADIOS-UsersManual-1.13.1.pdf>`_, chapter 8.2 for full details.

See ``adios_config -m`` for available compression methods and recompile ADIOS with further dependencies if needed.
Typically, ADIOS adds compressors during the ``configure`` step with options such as ``--with-zlib=<ZLIB_DIR>`` and ``--with-blosc=<BLOSC_DIR>``.

.. _usage-plugins-ADIOS-meta:

Meta Files
^^^^^^^^^^

Disabling on-the-fly meta (journal) file creation can improve output performance for large scale runs.
After your simulation finished, make sure to run ``bpmeta <theoretical-meta-fileName>`` on created ADIOS output.

You also need to create the meta file if you skipped on-the-fly creation in checkpointing and want to :ref:`restart from such a checkpoint <usage-plugins-checkpoint>` (with ADIOS as IO backend).

Example:

.. code-block:: bash

   ls simOutput/
   # bp  checkpoints  [...]

   ls simOutput/{bp,checkpoints}
   # simOutput/bp:
   #   simData_0.bp.dir simData_100.bp.dir [...]
   # simOutput/checkpoints:
   #   checkpoint_0.bp.dir checkpoint_2000.bp.dir

   cd simOutput/bp
   bpmeta simData_0.bp
   bpmeta simData_100.bp
   # [...]
   cd ../checkpoints
   bpmeta checkpoint_0.bp
   bpmeta checkpoint_2000.bp

   ls simOutput/{bp,checkpoints}
   # simOutput/bp:
   #   simData_0.bp simData_0.bp.dir
   #   simData_100.bp simData_100.bp.dir [...]
   # simOutput/checkpoints:
   #   checkpoint_0.bp checkpoint_0.bp.dir
   #   checkpoint_2000.bp checkpoint_2000.bp.dir

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
