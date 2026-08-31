Energy Histogram
================

The energy-histogram diagnostic records the kinetic-energy spectrum
of one species as a simple 1D histogram.
It is cheap to produce and easy to read,
which makes it the workhorse diagnostic for quick checks
("did the acceleration work? how hot is the plasma?").

.. literalinclude:: ../snippets/selected_topics/energy_histogram.py
   :language: python
   :start-after: histogram = EnergyHistogram
   :end-before: sim.run(

Parameters:

* ``species``:
  the species (or :ref:`filtered species <particle-filters>`) to record.
* ``period``:
  the :ref:`time steps <time-steps>` at which to write output.
* ``bin_count``:
  the number of histogram bins (must be positive).
* ``min_energy`` / ``max_energy``:
  the range of the histogram, **in keV**;
  ``min_energy`` must be smaller than ``max_energy``.

The output is a plain ASCII file
``simOutput/<species>_energyHistogram_<filter>.dat``
(one per species and filter):
the first line holds the bin edges in keV,
and each recorded time step appends one line of counts.
Because the file format is fixed,
you can post-process it directly with `numpy`_ and `matplotlib`_ --
see :ref:`Multiple simulations in a single script <python_package/foundations/defining_simulation:Multiple simulations in a single script>`
for a complete example of reading histograms from a parameter scan.

.. _numpy: https://numpy.org
.. _matplotlib: https://matplotlib.org
