.. _openpmd:

openPMD Output
==============

`openPMD <https://www.openpmd.org/>`__ is the general-purpose,
hierarchical data standard for particle and field data in computational physics.
It is the most flexible of the output formats PIConGPU offers:
fields, particles and arbitrarily derived quantities
are stored together in a single, self-describing file
that can be read by `openPMD-tools <https://github.com/openPMD/openPMD-api>`__,
`yt <https://yt-project.org/>`__ and similar analysis tools.

Three diagnostics write their data through openPMD,
all parameterized by the same :class:`~picongpu.picmi.diagnostics.OpenPMDConfig`:

``ParticleDump``
   Dumps all data of a species (position, momentum, weighting, ...)
   on the given time steps.

``NativeFieldDump``
   Dumps one of the native fields ``"E"``, ``"B"`` or ``"J"``
   (the fields PIConGPU solves for and the current).

``DerivedFieldDump``
   Deposition of an arbitrary particle quantity to the grid --
   e.g. a species' charge density, current density or kinetic energy --
   via a :ref:`particle functor <particle-functors>`.
   The field name is derived from the species, the optional filter
   and the functor name.

.. literalinclude:: ../../snippets/selected_topics/openpmd.py
   :language: python
   :start-at: @ParticleFunctor(name="kineticEnergy")
   :end-before: for config in sorted

Output files
------------

Each diagnostic's ``options`` (an ``OpenPMDConfig``)
determines the file it writes to:

* all openPMD files are written into ``simOutput/openPMD/``;
* the full file name is ``<file><infix>.<ext>``,
  with defaults ``infix="_%06T"`` (zero-padded iteration number)
  and ``ext="bp5"`` (the BP5 backend, the fastest one available;
  ``"h5"`` for HDF5 is also common);
* diagnostics that share an *equal* ``options``
  (and therefore the same file)
  are grouped into a single openPMD plugin --
  in the example above, the particle dump, the electric field
  and the derived field all use the default ``file="simData"``,
  so they are stored together in ``simData_%06T.bp5``.

For every group of shared options,
the input file generation writes an openPMD configuration file
to ``etc/`` of the setup directory (a TOML file,
referenced from the generated ``N.cfg`` via ``--openPMD.pluginConfig``).
It lists, per time step, which fields and particles are written:

.. literalinclude:: ../../snippets/selected_topics/openpmd.py
   :language: python
   :start-at: for config in sorted

.. note::

   The example above prints the generated configuration files.
   In a real setup directory, look for
   ``etc/openPMD_config_*.toml`` --
   the suffix is a hash of the configuration content,
   so the name changes when you change the diagnostics.

``OpenPMDConfig`` parameters
----------------------------

* ``file``:
  the file name base (required);
  an absolute path writes outside of ``simOutput/openPMD/``.
* ``infix``:
  inserted between file name and extension;
  the default ``"_%06T"`` makes each time step a separate file.
  Use ``infix=""`` to append to a single file (not recommended for parallel runs).
* ``ext``:
  the openPMD backend, ``"bp5"`` (default) or e.g. ``"h5"``.
* ``range``:
  restrict the dumped region to a cell range
  (a ``RangeSpec`` with one entry per dimension;
  each entry is ``None`` (full extent), a single cell index
  or a ``(start, stop)`` pair --
  e.g. ``RangeSpec((10, 20), None, None)`` dumps cells 10-20 in ``x`` only);
  the default is the full grid.
* ``data_preparation_strategy``:
  ``"mappedMemory"`` (default) or ``"doubleBuffer"``
  (lower memory, but the output of one step is only available after the next).
* ``backend_config``:
  additional openPMD backend options that are not exposed here,
  given as a typed :class:`~picongpu.picmi.diagnostics.OpenPMDBackendConfig`
  (or a plain ``dict``) and rendered into the plugin's ``backend_config``
  setting (see below).

Backend configuration
---------------------

``backend_config`` accepts a
:class:`~picongpu.picmi.diagnostics.OpenPMDBackendConfig`,
a typed model of the full openPMD backend schema
(`openPMD backend configuration <https://openpmd-api.readthedocs.io/en/latest/details/backendconfig.html>`__,
openPMD-api 0.17+).
It covers the backend-independent root (``backend``,
``iteration_encoding``, lazy-parsing hints, ``rank_table``),
the per-backend tables ``adios2``, ``hdf5``, ``json`` and ``toml``,
and the nested ADIOS2 engine/operators, HDF5 VFD/permanent filters
and JSON/TOML dataset/attribute sub-models.
Every option is optional and unset options are omitted,
so openPMD's own defaults apply.
Unknown keys are rejected with a validation error rather than
silently dropped, so typos surface immediately.

``rank_table`` takes the name of a host-name method as a string
(``"hostname"``, ``"mpi_processor_name"`` or ``"posix_hostname"``),
matching openPMD's own schema.

The per-dataset ``dataset`` list follows openPMD's pattern-matched form:
each entry is a ``{select, cfg}`` object, where ``cfg`` is **mandatory**
(use ``cfg = {}`` to accept the backend defaults) and ``select`` is optional.
The entry without ``select`` is the default configuration.

``resizable`` is deliberately **not** exposed: openPMD honours it only as a
per-``Dataset`` constructor option, so a backend-config key would be silently
ignored (the model rejects it with an explanatory error instead).

The example below selects the ADIOS2 backend, applies blosc compression
to every dataset through the default ``dataset`` entry,
then disables compression for the offset and patch datasets
through a pattern-matched entry
(``select`` is an egrep regex or a list of them;
the entry without ``select`` is the default,
and the first matching entry -- top-down -- wins):

.. literalinclude:: ../../snippets/selected_topics/openpmd_backend_config.py
   :language: python
   :start-after: # BEGIN-OPENPMD-BACKEND-CONFIG
   :end-before: # END-OPENPMD-BACKEND-CONFIG

A populated ``backend_config`` is rendered as a nested TOML table,
e.g. the ``adios2.dataset`` list becomes
``[[backend_config.adios2.dataset]]`` entries.
An explicit-but-empty ``OpenPMDBackendConfig()`` is normalised to
no configuration at all, so the generated TOML stays free of a
spurious empty ``backend_config`` key.

The same model is accepted by the ``openPMDBackendConfig`` parameter of
:class:`~picongpu.picmi.diagnostics.Binning`
(see :ref:`binning`); there it is serialised to a JSON string for the
binning plugin's own openPMD backend transport.
