:orphan:

.. only:: html

  .. image:: ../logo/pic_logo.svg

.. only:: latex

  .. image:: ../logo/pic_logo.pdf

*Particle-in-Cell Simulations for the Exascale Era*

PIConGPU is a fully relativistic, manycore, 3D3V and 2D3V particle-in-cell (PIC) code.
The PIC algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of electrons and ions in the plasma based on the Vlasov-Maxwell system of equations.

How to Read This Documentation
------------------------------

**Start here:** the :ref:`Quick Start <python_package/quickstart:Quick Start>`
takes you from zero to a running PIConGPU simulation in five steps using the
PICMI Python interface.

Pick the entry point that matches what you are looking for:

* **New to PIConGPU?**
  Work through the **Getting Started** chapter below, from installing the
  Python package to your first simulation, and continue with the
  :ref:`tutorial <python_package/tutorial:Tutorial: Setting up a simple LWFA>`.
* **Looking for a feature?**
  The **Physics & Features** chapter describes the physical models, and the
  **Using PIConGPU (PICMI)** chapter explains how to configure them through the
  PICMI interface, with the full class reference in the
  :ref:`API Documentation <python_package/api/index:API Documentation>`.
* **Running on a cluster or installing by hand?**
  The **Installation (HPC / from source)** chapter collects the manual C++
  dependency installation and the cluster profiles.
  For the Python interface, see
  :ref:`Installing the Python Package <python_package/install:Installing the Python Package>`.
* **Extending the C++ interface?**
  The **C++ Interface (advanced / legacy)** and **Development** chapters
  document the direct ``.param``/TBG interface and how to contribute.

Individual chapters build on the ones before them, but you can jump straight to
the section that answers your question.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of PIConGPU.
   If you are looking for the latest *development* version, `click here <https://picongpu.readthedocs.io/en/latest/>`_.


.. note::

   We are migrating our `wiki`_ to this manual, but some pages might still be missing.
   We also have an `official homepage`_ .

.. _wiki: https://github.com/ComputationalRadiationPhysics/picongpu/wiki
.. _official homepage: http://picongpu.hzdr.de

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   python_package/index

.. toctree::
   :caption: Using PIConGPU (PICMI)
   :maxdepth: 1
   :hidden:

   python_package/foundations/index
   python_package/selected_topics/index
   python_package/api/index

.. toctree::
   :caption: Physics & Features
   :maxdepth: 1
   :hidden:

   models/pic
   models/AOFDTD
   models/lasers
   models/total_field_scattered_field
   models/shapes
   models/LL_RR
   models/field_ionization
   models/collisional_ionization
   models/binary_collisions
   models/atomic_physics

.. toctree::
   :caption: Post-Processing
   :maxdepth: 2
   :hidden:

   postprocessing/python
   postprocessing/openPMD
   postprocessing/paraview
   usage/python_utils

.. toctree::
   :caption: C++ Interface (advanced / legacy)
   :maxdepth: 1
   :hidden:

   usage/basics
   usage/param
   usage/plugins
   usage/tbg
   usage/workflows
   usage/tests
   usage/examples
   tutorials/hemeraIn5min

.. toctree::
   :caption: Installation (HPC / from source)
   :maxdepth: 1
   :hidden:

   install/path
   install/instructions
   install/dependencies
   install/profile
   install/changelog.md

.. toctree::
   :caption: Expert
   :maxdepth: 1
   :hidden:

   expert/deviceOversubscription
   expert/signals
   usage/crosscompile
   usage/parameter_scans

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:

   usage/reference
   dev/CONTRIBUTING.md
   dev/docs/COMMIT.md
   dev/ci
   dev/repostructure
   dev/styleguide
   dev/sphinx
   dev/doxygen
   dev/clangtools
   dev/extending
   dev/picongpu
   dev/pmacc
   dev/py_postprocessing
   dev/debugging
   dev/doxyindex
   dev/PlantUML
   prgpatterns/lockstep
   testing/general
   testing/usage
   testing/structure
   testing/testbuilding
   testing/examples
