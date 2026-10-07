.. _species:

Species, Distributions and Layouts
==================================

Adding particles to a simulation takes three components:
a **species** (what the particles are),
a **distribution** (where they are placed and how they move), and a
**layout** (their positions *within* a cell).
They are passed to the simulation together:

.. code-block:: python

   sim = picmi.Simulation(
       ...,
       species=[ion, electron],
       layouts=[ion_layout, electron_layout],
   )

.. literalinclude:: ../snippets/selected_topics/species_distributions_layouts.py
   :language: python
   :start-at: plasma = picmi.UniformDistribution
   :end-before: simulation.write_input_file

Species
-------

A :class:`~picongpu.picmi.species.Species` describes one type of particle.
Its most important parameters are:

* ``name``:
  the name of the species (also used in the output files).
  If not given, the name is derived from the particle type.
* ``particle_type``:
  the physical identity of the particle.
  This is either an element symbol (``"H"``, ``"He"``, ``"C"``, ...),
  one of the predefined particle types
  (``"electron"``, ``"positron"``, ``"proton"``, ``"anti-proton"``, ``"photon"``, ...),
  or a custom particle of the form ``"other:<name>"``
  (which then requires an explicit ``name``).
  The mass and charge of known particle types are filled in automatically.
* ``charge_state``:
  the initial charge state of an ion
  (0 for neutral, 1 for singly ionized, ...).
  Only meaningful together with an element ``particle_type``.
  Ions that can be ionized further during the simulation
  (see :ref:`Interactions <python_package/selected_topics/interactions:Interactions>`)
  must specify their initial charge state explicitly.
* ``picongpu_fixed_charge``:
  for ion species that are *not* subject to ionization,
  this fixes the charge of all their particles for the entire simulation.
* ``mass`` / ``charge``:
  override the (element-)derived mass and charge in SI units.
  This is how you define custom particles.
* ``particle_shape``:
  the particle shape used for current/charge deposition.
  If left unset, it is inherited from
  :attr:`~picongpu.picmi.simulation.Simulation.particle_shape`
  and, if that too is unset, falls back to the PIConGPU default
  ``"quadratic"`` (i.e. TSC).
* ``method``:
  the particle pusher (default ``"Boris"``;
  ``"Vay"`` and ``"Higuera-Cary"`` are relativistic variants,
  ``"LLRK4"`` adds radiation reaction).
  For a pusher that changes over the course of the simulation, pass a
  :class:`~picongpu.picmi.species.CompositePusher` instead (see below).
* ``density_scale``:
  rescales the species' density relative to a shared profile
  (see below).

The per-species shape and pusher method default to the Simulation:
:attr:`~picongpu.picmi.simulation.Simulation.particle_shape`
is inherited by every species that does not set its own
(an explicit species ``particle_shape`` overrides it), while an unset
``method`` falls back to ``"Boris"``.
If neither the species nor the Simulation sets a shape,
PIConGPU uses its native default ``"quadratic"`` (TSC):

.. literalinclude:: ../snippets/selected_topics/species_shape_and_method.py
   :language: python
   :start-after: BEGIN-SPECIES_SHAPE
   :end-before: END-SPECIES_SHAPE

Step-dependent pushers
^^^^^^^^^^^^^^^^^^^^^^

Instead of one pusher for the whole run, a
:class:`~picongpu.picmi.species.CompositePusher` selects the active pusher per
time step. It maps **pre-called** ``TimeStepSpec`` instances -- i.e.
``TimeStepSpec[...]("steps")`` -- to pusher names; the **first matching
specification wins**, so the order of the mapping matters:

.. literalinclude:: ../snippets/selected_topics/composite_pusher.py
   :language: python
   :start-after: BEGIN-COMPOSITE-PUSHER
   :end-before: END-COMPOSITE-PUSHER

The example gives ``free-streaming`` for the first 100 steps and ``Boris``
thereafter. The specification keys reuse the full
:class:`~picongpu.picmi.diagnostics.TimeStepSpec` slice syntax (inclusive
bounds, ``::n``, negatives, open ends), and periodic/interleaved schedules such
as ``TimeStepSpec[::2]`` / ``TimeStepSpec[1::2]`` are supported. Keys must use
the ``"steps"`` unit; ``"seconds"`` cannot be used here because it would depend
on the time step size.

Every step in ``[0, max_steps)`` must be claimed by exactly one specification.
A gap is a hard error at input-file generation (there is no silent fallback), so
a catch-all specification is usually needed. If two adjacent intervals overlap
at a shared end -- for example ``TimeStepSpec[:5]("steps")`` and
``TimeStepSpec[5:]("steps")`` both claim step 5 -- a warning is emitted telling
you to make them disjoint (e.g. ``TimeStepSpec[:4]`` / ``TimeStepSpec[5:]``); the
earlier specification still wins. A single-entry composite is allowed and
behaves like the plain scalar pusher.

By default, every species is initialised **independently**: even when several
species happen to share the same distribution and layout, each one draws its
own in-cell positions.
To place several species *collectively* -- i.e. on exactly the same in-cell
positions, the standard way to build charge-neutral plasmas -- group them in a
:class:`~picongpu.picmi.multi_species.MultiSpecies`
(see :ref:`multi_species`).

.. _multi_species:

MultiSpecies: collective initialisation
---------------------------------------

A :class:`~picongpu.picmi.multi_species.MultiSpecies` is the explicit way to
request **collective (coordinated) initialisation**: all its members share one
``initial_distribution`` and one layout, and the whole group is placed with a
single density operation, so the members occupy exactly the same in-cell
positions -- and are therefore charge-neutral by construction, irrespective of
per-member momentum or temperature.

.. literalinclude:: ../snippets/selected_topics/multi_species.py
   :language: python
   :start-after: BEGIN-MULTI-SPECIES
   :end-before: END-MULTI-SPECIES

The whole :class:`~picongpu.picmi.multi_species.MultiSpecies` is one entry of
the simulation's ``species`` list, paired with a single layout in ``layouts``
(passed declaratively as shown above or via
:meth:`~picongpu.picmi.simulation.Simulation.add_species`). Its individual
members can be addressed by index or, if named, by name (e.g.
``multispecies["electrons"]``) and used elsewhere as needed.
The value at each position of ``proportions`` becomes the corresponding member's
``density_scale`` (its ``DensityRatio`` on the C++ level), so a
``proportions=[1.0, 1.0]`` ion/electron pair yields a neutral plasma.

.. note::

   Grouping is **structural**: the :class:`~picongpu.picmi.Simulation` stores
   each ``species`` entry as given. A whole ``MultiSpecies`` becomes one density
   operation covering all of its members; every standalone
   :class:`~picongpu.picmi.species.Species` entry becomes its own operation.
   There is therefore no implicit merging of look-alike species: to initialise
   species collectively, you must group them in a ``MultiSpecies``.

.. _distributions:

Particle Distributions
----------------------

A distribution describes *where* the particles of a species are placed
(their density profile) and *how they move* initially.
All distributions take

* ``rms_velocity``:
  a 3D vector of thermal velocity spreads in m/s
  (they are converted to a temperature internally), and
* ``directed_velocity``:
  a 3D vector of a collective drift velocity in m/s.

The available distributions are:

:class:`~picongpu.picmi.distribution.UniformDistribution`
   A constant density throughout the box (``density`` in m⁻³).

   .. note::

      The ``lower_bound``/``upper_bound`` and ``fill_in`` parameters
      are not supported: setting them to non-default values raises an
      ``UnsupportedFeatureError`` at input-file generation, while the
      density fills the entire simulation box. For sub-volume densities
      use ``AnalyticDistribution``, ``GaussianDistribution`` or
      ``FoilDistribution`` instead.

:class:`~picongpu.picmi.distribution.GaussianDistribution`
   A constant-density region with Gaussian ramps at the front and the rear
   of the box (in ``y`` direction):
   ``center_front``/``center_rear`` and ``sigma_front``/``sigma_rear``
   give the position and width of the ramps,
   ``power`` the exponent (2 is Gaussian, 4 and up super-Gaussian),
   ``factor`` the (negative) scaling of the ramps,
   and ``vacuum_front`` the vacuum in front of the profile.

:class:`~picongpu.picmi.distribution.FoilDistribution`
   A thin foil of constant ``thickness`` at position ``front``
   (perpendicular to ``y``),
   with optional exponential pre- and post-plasma ramps
   (``exponential_pre_plasma_length``/``_cutoff`` and
   ``exponential_post_plasma_length``/``_cutoff``).

:class:`~picongpu.picmi.distribution.CylindricalDistribution`
   A cylinder of ``radius`` around the axis ``cylinder_axis``
   through the point ``center_position``,
   with an optional exponential pre-plasma ramp
   (``exponential_pre_plasma_length``/``_cutoff``).

:class:`~picongpu.picmi.distribution.AnalyticDistribution`
   A density given by an analytic expression
   (see :ref:`the functors page <functors>`).

:class:`~picongpu.picmi.distribution.GaussianBunchDistribution`
   A finite 3D Gaussian particle bunch (the PICMI-standard
   ``GaussianBunchDistribution``).
   It is described by the number of physical particles
   ``n_physical_particles``, the per-axis RMS size ``rms_bunch_size``
   and the ``centroid_position`` of the bunch,
   plus an optional rigid ``centroid_velocity`` (given as ``gamma * v``)
   and a thermal ``rms_velocity``.
   The correlated ``velocity_divergence`` is not supported
   (a non-zero value raises an ``UnsupportedFeatureError``), and
   because the bunch is inherently three-dimensional it is rejected on a
   2D grid (an ``UnsupportedFeatureError`` at input-file generation).

   .. literalinclude:: ../snippets/selected_topics/gaussian_bunch.py
      :language: python
      :start-at: bunch = picmi.GaussianBunchDistribution
      :end-before: electrons = picmi.Species

:class:`~picongpu.picmi.distribution.FromFileDistribution`
   Loads the particles of a species from an external openPMD file
   (the PICMI-standard ``FromFileDistribution``).
   ``file_path`` points at the file and the optional ``iteration``
   selects the openPMD iteration to read (default ``0``).
   The file must provide ``position`` and ``positionOffset`` (the latter
   either a constant record component or the per-particle beginning-of-cell
   form; its ``unitSI`` is respected), ``momentum`` and ``weighting``.
   If the file additionally carries ``particlePatches`` in PIConGPU's
   cell-index layout it is read through the fast checkpoint reader;
   otherwise every rank reads the whole record and keeps only the particles
   inside its own domain, which is correct for any such file but slower and
   less memory-efficient.
   Because the particle positions come from the file, a from-file
   distribution must **not** be paired with a layout: passing one raises a
   ``ValueError``. It is currently rejected on a 2D grid, as is a
   dimensionality that does not match the simulation.

   .. literalinclude:: ../snippets/selected_topics/from_file.py
      :language: python
      :start-after: BEGIN-FROM-FILE
      :end-before: END-FROM-FILE

The reference density used to normalize the code units is
``simulation.picongpu_base_density`` (default ``1.0e25`` m⁻³).

Layouts
-------

The layout determines the positions of the particles *within* a cell.
It is given per species via the ``layouts`` list:

:class:`~picongpu.picmi.layout.PseudoRandomLayout`
  ``n_macroparticles_per_cell`` particles per cell at pseudo-random positions.
  This is the default choice for most simulations.

:class:`~picongpu.picmi.layout.GriddedLayout`
  A regular sub-grid of ``n_macroparticle_per_cell = [nx, ny, nz]``
  positions per cell (``nx * ny * nz`` particles per cell).
  Useful for well-resolved, low-noise configurations.

:class:`~picongpu.picmi.layout.OnePositionLayout`
  A single position per cell
  (``n_macroparticles_per_cell`` particles per cell, all at the same point,
  shifted by ``in_cell_offset`` in units of the cell size).

A species that starts out empty (``initial_distribution=None``,
e.g. electrons filled by ionization) gets the layout ``None``.
Giving a layout to a species without a distribution raises an error.
