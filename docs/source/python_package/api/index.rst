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
   :noindex:

Submodules
----------

The individual submodules hold the classes grouped by concern.
Most of them are re-exported in the top-level namespace above,
but you can also import them directly:

.. toctree::
   :maxdepth: 1
   :hidden:

   simulation
   grid
   solver
   species
   layout
   particle_functor
   distribution
   lasers
   applied_field
   interaction
   diagnostics

The ``pypicongpu`` package is the internal middle layer that the
``picmi`` frontend renders into, and the template layer renders it into the
C++ and configuration files. It is not part of the stable public API and is
intentionally not documented in this reference;
see :ref:`Foundations <python_package/foundations/index:Foundations>`
for how the layers fit together.
