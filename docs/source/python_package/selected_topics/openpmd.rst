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

.. literalinclude:: ../snippets/selected_topics/openpmd.py
   :language: python
   :start-after: def kinetic_energy
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

.. literalinclude:: ../snippets/selected_topics/openpmd.py
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
  a path to an additional openPMD backend configuration file
  for options that are not exposed here.
