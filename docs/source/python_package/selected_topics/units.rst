.. _units:

Units and Constants
===================

The PICMI frontend takes **SI units** for its inputs and converts them to
the code's internal units before they reach the simulation backend.
A small set of physical constants ships with the frontend as
``picongpu.picmi.constants`` (all taken from `scipy.constants
<https://docs.scipy.org/doc/scipy/reference/constants.html>`__):

* ``c``: speed of light,
* ``ep0``: vacuum permittivity (``epsilon_0``),
* ``mu0``: vacuum permeability,
* ``q_e``: elementary charge,
* ``m_e`` / ``m_p``: electron / proton mass,
* ``eV``: one electronvolt,
* ``keV``: one kiloelectronvolt (``1e3 * eV``),
* byte-size multipliers ``B``, ``kB``/``MB``/``GB`` (powers of 1000) and
  ``KiB``/``MiB``/``GiB`` (powers of 1024),
  e.g. for the :ref:`memory configuration <simulation_settings>`.

Use them to build the SI values the frontend expects:

.. literalinclude:: ../snippets/selected_topics/units_and_constants.py
   :language: python
   :start-at: from
   :end-before: print(

Common conventions
------------------

* **Length** (positions, bounds, cell size): metres.
* **Time** (``duration``, ``time_step_size``): seconds.
* **Density**: particles per cubic metre (m⁻³);
  normalized internally by ``simulation.picongpu_base_density``
  (default ``1.0e25`` m⁻³).
* **Momentum** (e.g. phase-space ranges): SI momenta (kg·m/s).
  One electron rest-mass momentum is ``m_e * c``
  (≈ ``2.73e-22`` kg·m/s).
* **Energy** (e.g. histogram ranges): SI joules;
  pass ``500 * constants.keV`` for 500 keV.
* **Velocity** (``rms_velocity``, ``directed_velocity``): m/s.
* **Laser amplitude**: give exactly one of the dimensionless ``a0``
  or the electric field ``E0`` in V/m.

The frontend converts as needed;
for example the energy-histogram bounds are divided by ``constants.keV``
and the phase-space momentum bounds by ``m_species * c``
before being handed to the C++ plugins.

Unit dimensions
---------------

Quantities built with :ref:`particle functors <particle-functors>`
can declare their physical unit with a
:class:`~picongpu.picmi.particle_functor.UnitDimension`,
so that the plugin can convert to the units it needs.
A unit dimension is a 7-component vector in the order
``L M T I Θ N J`` (length, mass, time, electric current, temperature,
amount of substance, luminous intensity) and can be built either from
single-letter keywords or from explicit exponents:

.. code-block:: python

   from picongpu.picmi.particle_functor import UnitDimension

   velocity = UnitDimension(L=1, M=0, T=-1)
   energy = UnitDimension(L=2, M=1, T=-2)

The predefined singletons ``L``, ``M``, ``T`` and ``I``
(from ``picongpu.picmi.particle_functor.unit_dimension``)
support multiplication and division, which makes physical expressions
readable:

.. code-block:: python

   from picongpu.picmi.particle_functor.unit_dimension import L, M, T

   momentum = M * L / T
   energy = M * L**2 / T**2
