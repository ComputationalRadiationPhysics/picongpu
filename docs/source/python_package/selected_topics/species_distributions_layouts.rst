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

When several species share the same distribution and layout,
they are placed at the same positions with their ``density_scale``
respected --
the standard way to build charge-neutral plasmas.

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
