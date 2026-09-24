.. _binning:

Binning
=======

The binning plugin is the most general particle diagnostic in PIConGPU:
it computes an *N-dimensional histogram of arbitrary particle properties*,
binning each particle into a cell of a user-defined multi-axis space
and depositing a user-defined quantity per particle into that cell.
Phase-space-like cuts, energy distributions, 2D emittance diagrams,
current-density maps and similar quantities are all special cases of this.

.. literalinclude:: ../../snippets/selected_topics/binning.py
   :language: python
   :start-at: binning = Binning
   :end-before: # BEGIN-BINNING-FILTER

Parameters:

* ``name``:
  the name of the binner (and of the output file).
* ``deposition_functor``:
  a :ref:`particle functor <particle-functors>` evaluated per particle
  and added to the bin the particle falls into --
  ``lambda p: 1.0`` counts particles.
* ``axes``:
  the list of :class:`~picongpu.picmi.diagnostics.BinningAxis`
  defining the dimensions of the histogram.
* ``species``:
  one species, a :ref:`filtered species <particle-filters>`
  or a list of both to bin.
* ``period``:
  the :ref:`time steps <time-steps>` at which to notify
  (i.e. bin) the particles; the default is every step.
* ``openPMDExt`` / ``openPMDInfix`` / ``openPMDBackendConfig`` / ``dumpPeriod``:
  the output format details;
  the output is written as openPMD to
  ``simOutput/binningOpenPMD/<name>_%06T.bp5`` by default.

Each ``BinningAxis`` combines

* a ``functor``:
  the particle property that forms this axis, and
* a ``bin_spec``:
  a :class:`~picongpu.picmi.diagnostics.BinSpec` with
  ``kind`` (``"linear"`` or ``"log"``), ``start``, ``stop`` and ``nsteps``;
  optionally
* ``name``:
  the axis name (defaults to the functor's name) and
* ``use_overflow_bins``:
  whether to count particles outside the ``[start, stop]`` range
  in two additional bins (default ``True``).

Functors and filters
--------------------

The axes and the deposition quantity of a binning are
:ref:`particle functors <particle-functors>`,
and a binning can be restricted to a
:ref:`filtered species <particle-filters>`.
Both are described in detail, together with the analytic density
distribution, on the
:ref:`functors page <functors>`.
