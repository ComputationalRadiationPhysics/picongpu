.. _usage-plugins-particleMerger:

Particle Merger
---------------

Merges macro particles that are close in phase space to reduce computational load.

.param file
^^^^^^^^^^^

In :ref:`particleMerging.param <usage-params-plugins>` is currently one compile-time parameter:

===================== ====================================================================================
Compile-Time Option   Description
===================== ====================================================================================
``MAX_VORONOI_CELLS`` Maximum number of active Voronoi cells per supercell.
                      If the number of active Voronoi cells reaches this limit merging events are dropped.
===================== ====================================================================================

.cfg file
^^^^^^^^^

============================================ ================================================================================================================
PIConGPU command line option                 Description
============================================ ================================================================================================================
``--<species>_merger.period``                The ouput periodicity of the plugin.
                                             A value of ``100`` would mean an output at simulation time step *0, 100, 200, ...*.
``--<species>_merger.minParticlesToMerge``   minimal number of macroparticles needed to merge the macroparticle collection into a single macroparticle.
``--<species>_merger.posSpreadThreshold``    Below this threshold of spread in position macroparticles can be merged [unit: cell edge length].
``--<species>_merger.absMomSpreadThreshold`` Below this absolute threshold of spread in momentum macroparticles can be merged [unit: :math:`m_{e-} \cdot c`].
                                             Disabled for ``-1`` (default).
``--<species>_merger.relMomSpreadThreshold`` Below this relative (to mean momentum) threshold of spread in momentum macroparticles can be merged [unit: none].
                                             Disabled for ``-1`` (default).
``--<species>_merger.minMeanEnergy``         minimal mean kinetic energy needed to merge the macroparticle collection into a single macroparticle [unit: keV].
============================================ ================================================================================================================

Notes
"""""

 - ``absMomSpreadThreshold`` and ``relMomSpreadThreshold`` are mutually exclusive
 - ``absMomSpreadThreshold`` is always given in [electron mass * speed of light]!

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations, but requires an extra particle attribute per species, ``voronoiCellId``.

Host
""""

no extra allocations.

Reference
^^^^^^^^^

The particle merger implements a macro particle merging algorithm based on:

Luu, P. T., Tueckmantel, T., & Pukhov, A. (2016).
Voronoi particle merging algorithm for PIC codes.
Computer Physics Communications, 202, 165-174.
