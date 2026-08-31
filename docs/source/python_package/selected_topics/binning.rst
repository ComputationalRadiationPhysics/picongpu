Binning
=======

The binning plugin is the most general particle diagnostic in PIConGPU:
it computes an *N-dimensional histogram of arbitrary particle properties*,
binning each particle into a cell of a user-defined multi-axis space
and depositing a user-defined quantity per particle into that cell.
Phase-space-like cuts, energy distributions, 2D emittance diagrams,
current-density maps and similar quantities are all special cases of this.

.. literalinclude:: ../snippets/selected_topics/binning.py
   :language: python
   :start-after: binning = Binning
   :end-before: sim.run(

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

.. _particle-functors:

Particle Functors
-----------------

A :class:`~picongpu.picmi.particle_functor.ParticleFunctor`
is a Python function of one (or two) arguments that describes
a particle property symbolically:

.. literalinclude:: ../snippets/selected_topics/binning.py
   :language: python
   :start-after: def gamma
   :end-before: grid = picmi.Cartesian3DGrid

The ``particle`` argument provides access to the particle's attributes
through ``particle.get("...")``:

* ``"position"``:
   a 3D vector; the keyword arguments ``origin``
   (``"total"`` (default), ``"cell"``, ``"local"``, ``"global"``,
   ``"moving_window"`` or ``"local_with_guards"``),
   ``precision`` (``"cell"`` (default) or ``"sub_cell"``)
   and ``unit`` (``"cell"`` (default), ``"pic"`` or ``"si"``)
   select the reference frame, the resolution and the units.
* ``"momentum"``:
  the 3D momentum in SI units (index it as ``px, py, pz = particle.get("momentum")``).
* ``"mass"``, ``"charge"``, ``"weighting"``:
  the particle's mass, charge and statistical weight.
* ``"gamma"``, ``"kinetic energy"``, ``"velocity"``:
  derived quantities, computed from mass and momentum.
* any further attribute name:
  rendered as a particle-attribute access in the generated code.

The function body is written with `sympy <https://www.sympy.org/>`__
expressions -- it is never executed in Python,
but *symbolically* (with dummy particle attributes)
and the resulting expression tree is compiled into the simulation binary,
where it is evaluated on the GPU.
Use ``return_type`` (e.g. ``int``) when the annotation of your function
is not enough, and ``unit_dimension``
(a :class:`~picongpu.picmi.particle_functor.UnitDimension`)
to declare the physical unit of the result.

.. _particle-filters:

Particle Filters
----------------

The same mechanism can be used to *select* particles instead of measuring them:
a :class:`~picongpu.picmi.particle_functor.ParticleFilter`
is a functor that must return a boolean,
and wrapping a species and a filter in a
:class:`~picongpu.picmi.particle_functor.FilteredSpecies`
gives a "species" that contains only the selected particles:

.. literalinclude:: ../snippets/selected_topics/binning.py
   :language: python
   :start-after: BEGIN-BINNING-FILTER
   :end-before: END-BINNING-FILTER

``FilteredSpecies`` is accepted everywhere a ``Species`` is
(phase space, energy histogram, particle dump, binning, ...).
The name of the filtered species is
``<species name>_<filter name>``,
which is also what you will find in the output files.

.. note::

   Deep dive:
   functors are compiled into the binary
   as ``ALPAKA_FN_ACC`` lambdas in the generated
   ``binningSetup.param`` / ``fileOutput.param``
   (see :ref:`the binning plugin documentation <usage-plugins-binningPlugin>`
   for the underlying C++ implementation).
   A functor that cannot be expressed in terms of the available
   particle attributes will fail at input-file generation
   with a message pointing at the offending expression.
