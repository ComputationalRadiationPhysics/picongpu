.. _usage-plugins-openPMD:

openPMD
-------

Stores simulation data such as fields and particles according to the `openPMD standard <https://github.com/openPMD/openPMD-standard>`_ using the `openPMD API <https://openpmd-api.readthedocs.io>`_.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`openPMD API <install-dependencies>` is compiled in.
If the openPMD API is found in version 0.13.0 or greater, PIConGPU will support streaming IO via openPMD.

.param file
^^^^^^^^^^^

The corresponding ``.param`` file is :ref:`fileOutput.param <usage-params-plugins>`.

One can e.g. disable the output of particles by setting:

.. code-block:: cpp

   /* output all species */
   using FileOutputParticles = VectorAllSpecies;
   /* disable */
   using FileOutputParticles = MakeSeq_t< >;

Particle filters used for output plugins, including this one, are defined in :ref:`particleFilters.param <usage-params-core>`.
Also see :ref:`common patterns of defining particle filters <usage-workflows-particleFilters>`.

.cfg file
^^^^^^^^^

Note that all the following command line parameters can *alternatively* be specified in a ``.toml`` configuration file.
See the next section for further information: `Configuring the openPMD plugin with a TOML configuration file>`

You can use ``--openPMD.period`` to specify the output period.
The base filename is specified via ``--openPMD.file``.
The openPMD API will parse the file name to decide the chosen backend and iteration layout:

* The filename extension will determine the backend.
* The openPMD will either create one file encompassing all iterations (group-based iteration layout) or one file per iteration (file-based iteration layout).
  The filename will be searched for a pattern describing how to derive a concrete iteration's filename.
  If no such pattern is found, the group-based iteration layout will be chosen.
  Please refer to the documentation of the openPMD API for further information.

In order to set defaults for these value, two further options control the filename:

* ``--openPMD.ext`` sets the filename extension.
  Possible extensions include ``bp`` for the ADIOS2 backend (default), ``h5`` for HDF5 and ``sst`` for Streaming via ADIOS2/SST.
  In case your openPMD API supports both ADIOS1 and ADIOS2, make sure that environment variable ``OPENPMD_BP_BACKEND`` is not set to ADIOS1.
* ``--openPMD.infix`` sets the filename pattern that controls the iteration layout, default is "_%06T" for a six-digit number specifying the iteration.
  Leave empty to pick group-based iteration layout.
  Since passing an empty string may be tricky in some workflows, specifying ``--openPMD.infix=NULL`` is also possible.

  Note that streaming IO requires group-based iteration layout in openPMD, i.e. ``--openPMD.infix=NULL`` is mandatory.
  If PIConGPU detects a streaming backend (e.g. by ``--openPMD.ext=sst``), it will automatically set ``--openPMD.infix=NULL``, overriding the user's choice.
  Note however that the ADIOS2 backend can also be selected via ``--openPMD.json`` and via environment variables which PIConGPU does not check.
  It is hence recommended to set ``--openPMD.infix=NULL`` explicitly.

Option ``--openPMD.source`` controls which data is output.
Its value is a comma-separated list of combinations of a data set name and a filter name.
A user can see all possible combinations for the current setup in the command-line help for this option.
Note that addding species and particle filters to ``.param`` files will automatically extend the number of combinations available.
By default all particles and fields are output.

For example, ``--openPMD.period 128 --openPMD.file simData --openPMD.source 'species_all'`` will write only the particle species data to files of the form ``simData_000000.bp``, ``simData_000128.bp`` in the default simulation output directory every 128 steps.
Note that this plugin will only be available if the openPMD API is found during compile configuration.

openPMD backend-specific settings may be controlled via two mechanisms:

* Environment variables.
  Please refer to the backends' documentations for information on environment variables understood by the backends.
* Backend-specific runtime parameters may be set via JSON in the openPMD API.
  PIConGPU exposes this via the command line option ``--openPMD.json``.
  Please refer to the openPMD API's documentation for further information.

The JSON parameter may be passed directly as a string, or by filename.
The latter case is distinguished by prepending the filename with an at-sign ``@``.
Specifying a JSON-formatted string from within a ``.cfg`` file can be tricky due to colliding escape mechanisms.
An example for a well-escaped JSON string as part of a ``.cfg`` file is found below.

.. literalinclude:: openPMD.cfg

PIConGPU further defines an **extended format for JSON options** that may alternatively used in order to pass dataset-specific configurations.
For each backend ``<backend>``, the backend-specific dataset configuration found under ``config["<backend>"]["dataset"]`` may take the form of a JSON list of patterns: ``[<pattern_1>, <pattern_2>, …]``.

Each such pattern ``<pattern_i>`` is a JSON object with key ``cfg`` and optional key ``select``: ``{"select": <pattern>, "cfg": <cfg>}``.

In here, ``<pattern>`` is a regex or a list of regexes, as used by POSIX ``grep -E``.
``<cfg>`` is a configuration that will be forwarded as-is to openPMD.

The single patterns will be processed in top-down manner, selecting the first matching pattern found in the list.
The regexes will be matched against the openPMD dataset path within the iteration (e.g. ``E/x`` or ``particles/.*/position/.*``), considering full matches only.

The **default configuration** is specified by omitting the ``select`` key.
Specifying more than one default is an error.
If no pattern matches a dataset, the default configuration is chosen if specified, or an empty JSON object ``{}`` otherwise.

A full example:

.. literalinclude:: openPMD_extended_config.json

The extended format is only available for configuration of the writing procedures.
The reading procedures (i.e. for restarting from a checkpoint) can be configured via ``--checkpoint.openPMD.jsonRestart``, e.g. see the example below for configuring the number of blosc decompression threads in ADIOS2.
Note that most sensible use cases for this command line option (including this example) require openPMD-api >= 0.15 (or a recent dev version until the 0.15 release).

.. literalinclude:: openPMD_restart_config.json

Two data preparation strategies are available for downloading particle data off compute devices.

* Set ``--openPMD.dataPreparationStrategy doubleBuffer`` for use of the strategy that has been optimized for use with ADIOS-based backends.
  The alias ``openPMD.dataPreparationStrategy adios`` may be used.
  This strategy requires at least 2x the GPU main memory on the host side.
  This is the default.
* Set ``--openPMD.dataPreparationStrategy mappedMemory`` for use of the strategy that has been optimized for use with HDF5-based backends.
  This strategy has a small host-side memory footprint (<< GPU main memory).
  The alias ``openPMD.dataPreparationStrategy hdf5`` may be used.

===================================== ====================================================================================================================================================
PIConGPU command line option          description
===================================== ====================================================================================================================================================
``--openPMD.period``                  Period after which simulation data should be stored on disk.
``--openPMD.source``                  Select data sources and filters to dump. Default is ``species_all,fields_all``, which dumps all fields and particle species.
                                      Only deterministic filters will be shown.
``--openPMD.range``                   Define a contiguous range of cells per dimension to dump. Each dimension is separated by a comma.
                                      A range is defined as ``[BEGIN:END)`` where out of range indices will be clipped.
                                      A single value e.g. ``10,:,:`` for a dimension will only dump a slice.
                                      Default is ``:,:,:``, which dumps all cells.
``--openPMD.file``                    Relative or absolute openPMD file prefix for simulation data. If relative, files are stored under ``simOutput``.
``--openPMD.ext``                     openPMD filename extension (this controls thebackend picked by the openPMD API).
``--openPMD.infix``                   openPMD filename infix (use to pick file- or group-based layout in openPMD). Set to NULL to keep empty (e.g. to pick group-based iteration layout).
``--openPMD.json``                    Set backend-specific parameters for openPMD backends in JSON format. Used in writing procedures.
``--checkpoint.openPMD.jsonRestart``  Set backend-specific parameters for openPMD backends in JSON format for restarting from a checkpoint.
``--openPMD.dataPreparationStrategy`` Strategy for preparation of particle data ('doubleBuffer' or 'mappedMemory'). Aliases 'adios' and 'hdf5' may be used respectively.
``--openPMD.toml``                    Alternatively configure the openPMD plugin via a TOML file (see below).
``--openPMD.particleIOChunkSize``     Particle data will be written in chunks of the given size. unit: MiB
                                      The memory footprint on the host side is reduced compared to writing all particles with one write call.
                                      Currently the memory footprint is only reduced if bp5 is used, HDF5 and bp4 does not support partial data dump.
                                      If your system host memory is small compared to the accelerator memory you should use ``--openPMD.dataPreparationStrategy mappedMemory`` to have
                                      a smaller host memory footprint.
                                      The memory footprint required to dump fields will not be affected by this parameter.
===================================== ====================================================================================================================================================

.. note::

   This plugin is a multi plugin.
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   Each plugin instance requires that either ``--openPMD.period`` XOR ``--openPMD.toml`` is defined.
   If ``--openPMD.toml`` is defined, the rest of the configuration for this instance is done via the specified TOML file, no other command line parameters may be passed.

   In the case where an optional parameter with a default value is explicitly defined, the parameter will always be passed to the instance of the multi plugin where the parameter is not set.
   e.g.

   .. code-block:: bash

      --openPMD.period 128 --openPMD.file simData1 --openPMD.source 'species_all'
      --openPMD.period 1000 --openPMD.file simData2 --openPMD.source 'fields_all' --openPMD.ext h5

   creates two plugins:

   #. dump all species data each 128th time step, use HDF5 backend.
   #. dump all field data each 1000th time step, use the default ADIOS backend.

Backend-specific notes
^^^^^^^^^^^^^^^^^^^^^^

ADIOS2
======

Streaming with the MPI-based SST implementation
"""""""""""""""""""""""""""""""""""""""""""""""

The `SST engine <https://adios2.readthedocs.io/en/latest/engines/engines.html#sst-sustainable-staging-transport>`_ (sustainable staging transport) of ADIOS2 has a number of possible backends, including an MPI data transport (still in development at the time of writing this 2023-01-06).
It internally uses MPI to set up communcation between data producer (PIConGPU) and consumer via ``MPI_Open_Port()`` and ``MPI_Comm_Accept()``.

Use of this engine requires availability of threaded MPI, which can be activated in PIConGPU via the environment variable ``PIC_USE_THREADED_MPI``.
Its possible values include ``"MPI_THREAD_SINGLE"``, ``"MPI_THREAD_FUNNELED"``, ``"MPI_THREAD_SERIALIZED"`` and ``"MPI_THREAD_MULTIPLE"``.
The specified value determines the ``int required`` parameter that will be used in the initialization of MPI via ``int MPI_Init_thread( int *argc, char ***argv, int required, int *provided )``.
For support of MPI-based SST, the variable must be specified as ``PIC_USE_THREADED_MPI=MPI_THREAD_MULTIPLE``.

Additionally, `a workaround <https://github.com/ornladios/ADIOS2/blob/770d852ac1e5d2faff0b9e62cb8da43d448cca67/docs/user_guide/source/advanced/ecp_hardware.rst>`_ is required in order to let PIConGPU cleanly terminate after using this engine on ECP hardware.
This workaround can be activated on such systems via ``PIC_WORKAROUND_CRAY_MPI_FINALIZE=1``.

HDF5
====


Chunking
""""""""

By default, the openPMD-api uses a heuristic to automatically set an appropriate `dataset chunk size <https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/>`_.
In combination with some MPI-IO backends (e.g. ROMIO), this has been found to cause crashes.
To avoid this, PIConGPU overrides the default choice and deactivates HDF5 chunking in the openPMD plugin.

If you want to use chunking, you can ask for it via the following option passed in ``--openPMD.json``:

.. code-block:: json

  {
    "hdf5": {
      "dataset": {
        "chunks": "auto"
      }
    }
  }

Performance tuning on Summit
""""""""""""""""""""""""""""

In order to avoid a performance bug for parallel HDF5 on the ORNL Summit compute system, a specific version of ROMIO should be chosen and performance hints should be passed:

.. code-block:: bash

  > export OMPI_MCA_io=romio321
  > export ROMIO_HINTS=./my_romio_hints
  > cat << EOF > ./my_romio_hints
  romio_cb_write enable
  romio_ds_write enable
  cb_buffer_size 16777216
  cb_nodes <number_of_nodes>
  EOF

Replace ``<number_of_nodes>`` with the number of nodes that your job uses.
These settings are applied automatically in the Summit templates found in ``etc/picongpu/summit-ornl``.
For further information, see the `official Summit documentation <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#slow-performance-using-parallel-hdf5-resolved-march-12-2019>`_ and `this pull request for WarpX <https://github.com/ECP-WarpX/WarpX/pull/2495>`_.


Performance
^^^^^^^^^^^

On the Summit compute system, specifying ``export IBM_largeblock_io=true`` disables data shipping, which leads to reduced overhead for large block write operations.
This setting is applied in the Summit templates found in ``etc/picongpu/summit-ornl``.

Configuring the openPMD plugin with a TOML configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The openPMD plugin can alternatively be configured by using a ``.toml`` configuration file.
Note the inline comments for a description of the used schema:

.. literalinclude:: openPMD.toml

The location of the ``.toml`` file on the filesystem is specified via ``--openPMD.toml``.
If using this parameter, no other parameters must be specified.
If another parameter is specified, the openPMD plugin will notice and abort.


Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations.

Host
""""

As soon as the openPMD plugin is compiled in, one extra ``mallocMC`` heap for the particle buffer is permanently reserved.
During I/O, particle attributes are allocated one after another.
Using ``--openPMD.dataPreparationStrategy doubleBuffer`` (default) will require at least 2x the GPU memory on the host side.
For a smaller host side memory footprint (<< GPU main memory) pick ``--openPMD.dataPreparationStrategy mappedMemory``.
In addition to using the mapped-memory strategy, you can reduce the memory footprint for particle IO to less than ``2*N`` MiB ``--openPMD.particleIOChunkSize N``.

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.

Experimental: Asynchronous writing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implements (part of) the workflow described in section 2 of `this paper <https://arxiv.org/abs/2107.06108>`_.
Rather than writing data to disk directly from PIConGPU, data is streamed via ADIOS2 SST (sustainable streaming transport) to a separate process running asynchronously to PIConGPU.
This separate process (``openpmd-pipe``) captures the stream and writes it to disk.
``openpmd-pipe`` is a Python-based script that comes with recent development versions of openPMD-api (commit bf5174da20e2aeb60ed4c8575da70809d07835ed or newer).
A template is provided under ``etc/picongpu/summit-ornl/gpu_batch_pipe.tpl`` for running such a workflow on the Summit supercompute system.
A corresponding single-node runtime configuration is provided for the KelvinHelmholtz example under ``share/picongpu/examples/KelvinHelmholtz/etc/picongpu/6_pipe.cfg`` (can be scaled up to multi-node).
It puts six instances of PIConGPU on one node (one per GPU) and one instance of ``openpmd-pipe``.

Advantages:

* Checkpointing and heavy-weight output writing can happen asynchronously, blocking the simulation less.
* Only one file per node is written, implicit node-level aggregation of data from multiple instances of PIConGPU to one instance of ``openpmd-pipe`` per node.
  ADIOS2 otherwise also performs explicit node-level data aggregation via MPI; with this setup, ADIOS2 can (and does) skip that step.
* This setup can serve as orientation for the configuration of other loosely-coupled workflows.

Drawbacks:

* Moving all data to another process means that enough memory must be available per node to tolerate that.
  One way to interpret such a setup is to use idle host memory as a buffer for asynchronous writing in the background.
* Streaming data distribution strategies are not yet mainlined in openPMD-api, meaning that ``openpmd-pipe`` currently implements a simple ad-hoc data distribution:
  Data is distributed by simply dividing each dataset into n equal-sized chunks where n is the number of reading processes.
  In consequence, communication is currently not strictly intra-node.
  ADIOS2 SST currently relies solely on inter-node communication infrastructure anyway, and performance differences are hence expected to be low.

Notes on the implementation of a proper template file:

* An asynchronous job can be launched by using ordinary Bash syntax for asynchronous launching of processes (``this_is_launched_asynchronously & and_this_is_not;``).
  Jobs must be waited upon using ``wait``.
* Most batch systems will forward all resource allocations of a batch script to launched parallel processes inside the batch script.
  When launching several processes asynchronously, resources must be allocated explicitly.
  This includes GPUs, CPU cores and often memory.
* This setup is currently impossible to implement on the HZDR Hemera cluster due to a wrong configuration of the Batch system.
