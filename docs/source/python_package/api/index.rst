API Documentation
=================

This page lists the public API of the PIConGPU Python package --
the :mod:`picongpu.picmi` module that you import as
``from picongpu import picmi``.
It is generated directly from the package with Sphinx ``autodoc``,
so it stays in sync with the code.

The :ref:`foundations <python_package/foundations/index:Foundations>`
and :ref:`selected topics <python_package/selected_topics/index:Selected Topics>`
pages explain *how* to use these classes in practice;
this page is the reference for *what* is available.

The top-level :mod:`picongpu.picmi` namespace re-exports the classes you
will use most:

.. automodule:: picongpu.picmi
   :members:
   :imported-members:
   :undoc-members:

Submodules
----------

The individual submodules hold the classes grouped by concern.
Most of them are re-exported in the top-level namespace above,
but you can also import them directly:

* :mod:`picongpu.picmi.diagnostics` -- the output/diagnostic plugins
* :mod:`picongpu.picmi.lasers` -- the laser pulse classes
* :mod:`picongpu.picmi.distribution` -- the particle distribution classes
* :mod:`picongpu.picmi.interaction` -- ionization, collision and synchrotron
* :mod:`picongpu.picmi.particle_functor` -- particle functors and filters
* :mod:`picongpu.picmi.grid` -- the simulation grid
* :mod:`picongpu.picmi.solver` -- the field solver
* :mod:`picongpu.picmi.species` -- particle species
* :mod:`picongpu.picmi.layout` -- particle layouts
* :mod:`picongpu.picmi.simulation` -- the :class:`Simulation` object

The ``pypicongpu`` package is the internal middle layer that the
``picmi`` frontend renders into; it is not part of the stable public API.
Its own (auto-generated) reference is included elsewhere in these
documentation.
