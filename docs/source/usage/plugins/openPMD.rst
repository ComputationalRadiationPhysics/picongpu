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

.cfg file
^^^^^^^^^

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
  Possible extensions include ``bp`` for the ADIOS backends (default), ``h5`` for HDF5 and ``sst`` for Streaming via ADIOS2/SST.
  If the openPMD API has been built with support for the ADIOS1 and ADIOS2 backends, ADIOS2 will take precedence over ADIOS1.
  This behavior can be overridden by setting the environment variable ``OPENPMD_BP_BACKEND=ADIOS1``.
* ``--openPMD.infix`` sets the filename pattern that controls the iteration layout, default is "_06T" for a six-digit number specifying the iteration.
  Leave empty to pick group-based iteration layout.
  Since passing an empty string may be tricky in some workflows, specifying ``--openPMD.infix=NULL`` is also possible.

  Note that streaming IO requires group-based iteration layout in openPMD, i.e. ``--openPMD.infix=NULL`` is mandatory.
  If PIConGPU detects a streaming backend (e.g. by ``--openPMD.ext=sst``), it will automatically set ``--openPMD.infix=NULL``, overriding the user's choice.
  Note however that the ADIOS2 backend can also be selected via ``--openPMD.json`` and via environment variables which PIConGPU does not check.
  It is hence recommended to set ``--openPMD.infix=NULL`` explicitly.

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
``--openPMD.source``                  Select data sources to dump. Default is ``species_all,fields_all``, which dumps all fields and particle species.
``--openPMD.compression``             Legacy parameter to set data transform compression method to be used for ADIOS1 backend until it implements setting compression from JSON config.
``--openPMD.file``                    Relative or absolute openPMD file prefix for simulation data. If relative, files are stored under ``simOutput``. 
``--openPMD.ext``                     openPMD filename extension (this controls thebackend picked by the openPMD API).
``--openPMD.infix``                   openPMD filename infix (use to pick file- or group-based layout in openPMD). Set to NULL to keep empty (e.g. to pick group-based iteration layout).
``--openPMD.json``                    Set backend-specific parameters for openPMD backends in JSON format.
``--openPMD.dataPreparationStrategy`` Strategy for preparation of particle data ('doubleBuffer' or 'mappedMemory'). Aliases 'adios' and 'hdf5' may be used respectively.
===================================== ====================================================================================================================================================

.. note::

   This plugin is a multi plugin. 
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   In the case where an optional parameter with a default value is explicitly defined, the parameter will always be passed to the instance of the multi plugin where the parameter is not set.
   e.g.

   .. code-block:: bash

      --openPMD.period 128 --openPMD.file simData1 --openPMD.source 'species_all' 
      --openPMD.period 1000 --openPMD.file simData2 --openPMD.source 'fields_all' --openPMD.ext h5

   creates two plugins:

   #. dump all species data each 128th time step, use HDF5 backend.
   #. dump all field data each 1000th time step, use the default ADIOS backend.

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

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.
