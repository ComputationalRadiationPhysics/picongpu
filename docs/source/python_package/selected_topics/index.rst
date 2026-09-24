Selected Topics
===============

This chapter expands on selected features of our PICMI dialect.
It will help you to get the most out of the features of PIConGPU
and set up your simulation precisely as you intend it.
It is not about the mechanics of actually running your simulation
or other concepts behind PIConGPU Python package.
See :ref:`Foundations <python_package/foundations/index:Foundations>` for all the details on that.
If you are just getting started,
you will probably have the best experience from following our :ref:`Quick Start <python_package/quickstart:Quick Start>` guide first.
You can return here afterwards to expand your knowledge on specific aspects.

The sections below are grouped by the kind of entity they describe:
the building blocks of a simulation
(grids, solvers, lasers, species, distributions, layouts),
the symbolic machinery used to express arbitrary quantities
(functors and filters),
the diagnostics (plugins) that extract data,
and the physics interactions between particles.
The remaining pages are deep dives into cross-cutting features.

.. toctree::
   :maxdepth: 2
   :hidden:

   grids_and_solvers
   lasers
   species_distributions_layouts
   functors
   simulation_settings
   diagnostics/index
   interactions
   units
   serialization
   custom_input
   hpc_submission
