Phase Space
===========

The phase-space diagnostic records a 2D histogram of a spatial coordinate
against a momentum coordinate for one species --
the standard tool to inspect the structure of your particle population
(e.g. the wake in a laser wakefield acceleration run).

.. literalinclude:: ../snippets/selected_topics/phase_space.py
   :language: python
   :start-after: phase_space = PhaseSpace
   :end-before: sim.run(

Parameters:

* ``species``:
  the species (or :ref:`filtered species <particle-filters>`) to record.
* ``period``:
  the :ref:`time steps <time-steps>` at which to write output.
* ``spatial_coordinate``:
  one of ``"x"``, ``"y"``, ``"z"`` -- the position axis of the histogram.
* ``momentum_coordinate``:
  one of ``"px"``, ``"py"``, ``"pz"`` -- the momentum axis.
* ``min_momentum`` / ``max_momentum``:
  the range of the momentum axis, in SI units (kg·m/s);
  ``min_momentum`` must be smaller than ``max_momentum``.

The output is written as `openPMD <https://www.openpmd.org/>`__ files
(HDF5 by default) into the ``simOutput/phaseSpace/`` directory;
one file per species, coordinate pair and time step.
The spatial extent of the recorded region follows the particles,
so in a moving-window simulation the covered region changes with time.

.. note::

   The momentum range is fixed for the whole run.
   Choose it wide enough for the highest momenta you expect,
   or you will miss them.
