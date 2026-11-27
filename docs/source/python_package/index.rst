Getting Started
===============

.. sectionauthor:: Julian Lenz

This chapter gets you from zero to a running PIConGPU simulation
through the PICMI Python interface.
It contains

* the :ref:`Quick Start <python_package/quickstart:Quick Start>` -- the five
  commands that install the Python package, build a runtime configuration,
  install dependencies, download an example and run it,
* the :ref:`tutorial <python_package/tutorial:Tutorial: Setting up a simple LWFA>`
  -- a complete, working laser-wakefield-acceleration example,
* the :ref:`low-level migration guide <legacy_migration>` -- maps the files of a
  low-level ``.param``/``.cfg`` setup to the PICMI classes that can produce
  their content.

Once you have run your first simulation, continue with
:ref:`Foundations <python_package/foundations/index:Foundations>`
for the core concepts behind the interface and
:ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`
for particular features.

.. toctree::
   :maxdepth: 1

   quickstart
   tutorial
   troubleshooting
   legacy_migration
   conquering_new_cluster/index
