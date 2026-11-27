.. _usage-python-utils-plugins:

Plugin Output Visualizers (matplotlib)
--------------------------------------

.. sectionauthor:: Julian Lenz

For each plugin that produces plottable output, the package ships a matplotlib visualizer under ``picongpu.extra.plugins.plot_mpl``.
All of them derive from :class:`~picongpu.extra.plugins.plot_mpl.base_visualizer.Visualizer` and follow the same usage pattern: construct one with one or more run directories and a matplotlib axes object, then call ``visualize`` with plugin-specific keyword arguments such as ``iteration``, ``time``, ``species`` and ``species_filter``.

.. code:: python

   from picongpu.extra.plugins.plot_mpl import EnergyHistogramMPL, PhaseSpaceMPL
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(1, 1)

   # a single run: (label, path to the run directory, i.e. the path before ``simOutput/``)
   energy_histogram = EnergyHistogramMPL(("sim", "/path/to/run"), ax)
   energy_histogram.visualize(iteration=200, species="e")

   # several runs at once, each with its own label and colour
   phase_space = PhaseSpaceMPL([("sim1", "/path/to/run1"), ("sim2", "/path/to/run2")], ax)
   phase_space.visualize(iteration=200, species="e", ps="ypx")

   plt.show()

A ``Visualizer`` instance can be reused: calling ``visualize`` again with a new iteration updates the existing artists instead of creating a new plot.
The run directories are resolved by the matching data reader in ``picongpu.extra.plugins.data``, which uses :ref:`FindTime <usage-python-utils-find-time>` to accept a simulation ``time`` as an alternative to an ``iteration``.

The following visualizers are available; their accepted keyword arguments are documented in the ``visualize`` method and on the corresponding plugin page.

============================================= ====================================== ==================================================
Module                                        Public alias in ``plot_mpl``           Plugin documentation
============================================= ====================================== ==================================================
``energy_histogram_visualizer``               ``EnergyHistogramMPL``                 :ref:`usage-plugins-energyHistogram`
``energy_waterfall_visualizer``               ``EnergyWaterfallMPL``                 :ref:`usage-plugins-energyHistogram`
``emittance_evolution_visualizer``            ``EmitanceEvolutionMPL``               :ref:`usage-plugins-sliceEmittance`
``phase_space_visualizer``                    ``PhaseSpaceMPL``                      :ref:`usage-plugins-phaseSpace`
``png_visualizer``                            ``PNGMPL``                             :ref:`usage-plugins-PNG`
``slice_emittance_visualizer``                ``SliceEmitanceMPL``                   :ref:`usage-plugins-sliceEmittance`
``slice_emittance_waterfall_visualizer``      ``SliceEmitanceWaterfallMPL``          :ref:`usage-plugins-sliceEmittance`
``transition_radiation_visualizer``           ``TransitionRadiationMPL``             :ref:`usage-plugins-transitionRadiation`
============================================= ====================================== ==================================================

.. note::

   The emittance aliases are spelled ``Emitance...`` (without the second ``t``); this historical spelling is part of the public API.

Several of the visualizer modules can also be run as scripts.
They typically accept ``-p/--path``, ``-i/--iteration``, ``-s/--species`` and ``-f/--filter``; the phase-space, PNG and transition-radiation visualizers add plugin-specific options (``-m`` for the phase-space selection, ``-a``/``-o`` for the PNG axis and slice offset, ``-t`` for the figure type).
See the individual plugin pages for the exact invocation.

The base class, together with the interface a new visualizer has to implement, is described in :ref:`development-pytools`.

EnergyHistogramMPL
~~~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.energy_histogram_visualizer.Visualizer
   :noindex:
   :members:

EnergyWaterfallMPL
~~~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.energy_waterfall_visualizer.Visualizer
   :noindex:
   :members:

EmitanceEvolutionMPL
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.emittance_evolution_visualizer.Visualizer
   :noindex:
   :members:

PhaseSpaceMPL
~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.phase_space_visualizer.Visualizer
   :noindex:
   :members:

PNGMPL
~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.png_visualizer.Visualizer
   :noindex:
   :members:

SliceEmitanceMPL
~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.slice_emittance_visualizer.Visualizer
   :noindex:
   :members:

SliceEmitanceWaterfallMPL
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.slice_emittance_waterfall_visualizer.Visualizer
   :noindex:
   :members:

TransitionRadiationMPL
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: picongpu.extra.plugins.plot_mpl.transition_radiation_visualizer.Visualizer
   :noindex:
   :members:
