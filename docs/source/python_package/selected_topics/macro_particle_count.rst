Macro-Particle Count
====================

The macro-particle-count diagnostic counts the total number of
macro-particles of a species and writes the count to a plain file.
It is one of the cheapest diagnostics available and a very useful
debugging tool:
it immediately tells you whether your ionization model produces particles,
whether particles are being created or lost,
and how the population evolves over time.

.. literalinclude:: ../snippets/selected_topics/macro_particle_count.py
   :language: python
   :start-after: counter = MacroParticleCount
   :end-before: sim.run(

Parameters:

* ``species``:
  the species to count.
* ``period``:
  the :ref:`time steps <time-steps>` at which to write output.

The output is a plain ASCII file
``simOutput/<species>_macroParticlesCount.dat``
(one line per recorded time step).
