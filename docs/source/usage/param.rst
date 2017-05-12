.. _usage-params:

.param Files
============

.. sectionauthor:: Axel Huebl

Parameter files, ``*.param`` placed in ``include/simulation_defines/param/`` are used to set all **compile-time options** for a PIConGPU simulation.
This includes most fundamental options such as numerical solvers, floating precision, memory usage due to attributes and super-cell based algorithms, density profiles, initial conditions etc.

Rationale
---------

High-performance hardware comes with a lot of restrictions on how to use it, mainly memory, control flow and register limits.
In order to create an efficient simulation, PIConGPU compiles to **exactly** the numerical solvers (kernels) and physical attributes (fields, species) for the setup you need to run, which will furthermore be specialized for a specific hardware.

This comes at a small cost: when one of those settings is changed, you need to recompile.
Nevertheless, wasting about 5 minutes compiling on a single node is nothing compared to the time you save *at scale*!

All options that are less or non-critical for runtime performance, such as specific ranges observables in :ref:`plugins <usage-plugins>` or how many nodes shall be used, can be set in :ref:`run time configuration files (*.cfg) <usage-tbg>` and do not need a recompile when changed.

Files and Their Usage
---------------------

If you use our ``pic-configure`` :ref:`script wrappers <usage-basics>`, you do not need to set *all* available parameter files since we will add the missing ones with *sane defaults*.
Those defaults are:

* a standard, single-precision, well normalized PIC cycle suitable for relativistic plasmas
* no external forces (no laser, no initial density profile, no background fields, etc.)

All Files
---------

When setting up a simulation, it is recommended to adjust ``.param`` files in the following order:

.. toctree::
   :maxdepth: 2

   param/core
   param/memory
   param/extensions
   param/plugins
   param/misc
