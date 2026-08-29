Radiation
=========

The radiation plugin computes the electromagnetic radiation field
emitted by the particles of a species
(synchrotron radiation, transition radiation, laser-plasma sources, ...).
The field is evaluated on a set of *virtual observers*
-- directions on the unit sphere --
and accumulated into spectra that are written to openPMD files.

.. literalinclude:: ../snippets/selected_topics/radiation.py
   :language: python
   :start-after: radiation = Radiation
   :end-before: sim.run(

The plugin requires a bit more setup than the other diagnostics,
because you must tell it *where* to look.

``species``
   The species (or list of species) whose radiation is computed.
   The plugin registers the necessary particle attributes (previous-step
   momentum, and a radiation mask when a ``gamma_filter_threshold`` is set)
   automatically.

``period``
   The :ref:`time steps <time-steps>` at which output is written.
   Note that the plugin **cannot produce output at time step 0**
   (it needs a few steps of particle history),
   so the first entry of the period must be ``>= 2``.

``observer``
   A ``RadiationObserverConfiguration`` (from
   ``picongpu.pypicongpu.output.radiation``) with

   * ``N_observer``:
     the number of observation directions, and
   * ``index_to_direction``:
     a function that maps the observer index (a `sympy <https://www.sympy.org/>`__
     symbol in ``[0, N_observer)``) to a 3D direction vector.
     The direction must have a well-defined (non-zero) norm;
     it is normalized internally.
     The example above distributes the observers over the unit sphere.

``num_accumulation_steps``
   How often the accumulated radiation is dumped to disk
   (``0`` -- the default -- never dumps).

``total_radiation`` / ``folder_total_rad``
   When set, the spectrum summed from the start of the simulation
   to the current step is stored (in the folder ``totalRad`` by default).

``last_radiation`` / ``folder_last_rad``
   When set, the spectrum summed between the previous dump and the
   current step is stored (in the folder ``lastRad`` by default),
   which is convenient for following the temporal evolution of the source.

``start`` / ``end``
   The time steps between which the radiation is accumulated
   (default: from step 2 to the end).

``gamma_filter_threshold``
   Only particles with a Lorentz factor above the threshold contribute;
   a cheap way to focus on the relativistic tail of the distribution.

The remaining parameters (frequency range ``frequencies``,
``nyquist_factor``, ``form_factor``, window function, ``rad_per_gpu``, ...)
control the spectral resolution and the particle-charge form factor;
see the ``RadiationConfiguration``/``RadiationPluginConfig`` model
in ``picongpu.pypicongpu.output.radiation`` for their full documentation.

.. note::

   The plugin is computationally heavy
   (it evaluates the vector potential on every observer direction).
   Use a moderate ``N_observer`` and a coarse frequency grid
   for exploration runs,
   and increase the resolution only for the final production run.
