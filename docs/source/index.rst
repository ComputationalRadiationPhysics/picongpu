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
  Work through the :ref:`Getting Started <python_package/index:Getting Started>`
  chapter below, from installing the Python package to your first simulation,
  and continue with the
  :ref:`tutorial <python_package/tutorial:Tutorial: Setting up a simple LWFA>`.
* **Looking for a feature?**
  The :ref:`Physics & Features <models/pic:The Particle-in-Cell Algorithm>`
  chapter describes the physical models, and the
  :ref:`Using PIConGPU (PICMI) <python_package/foundations/index:Foundations>`
  chapter explains how to configure them through the PICMI interface, with the
  full class reference in the
  :ref:`API Documentation <python_package/api/index:API Documentation>`.
* **Migrating from legacy** ``.param`` **file setups to PICMI?**
  The :ref:`Foundations <python_package/foundations/index:Foundations>` chapter
  introduces the PICMI interface, and the
  :ref:`Legacy C++ interface (advanced) <usage-index>` chapter still documents
  the legacy ``.param``/TBG setup you are coming from, so you can map an
  existing configuration onto a PICMI input script.
* **Developing the C++ core?**
  The :ref:`Legacy C++ interface (advanced) <usage-index>` and
  :ref:`Development <dev/CONTRIBUTING:How to Participate as a Developer>`
  chapters document the direct ``.param``/TBG interface and how to contribute.
  The legacy chapter also contains the
  :ref:`installation instructions <install/path:Introduction>` for the manual
  C++ dependency installation and the cluster profiles.
  For the Python interface, see
  :ref:`Installing the Python Package <python_package/selected_topics/install:Installing the Python Package>`.

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

   postprocessing/openPMD
   postprocessing/paraview
   usage/python_utils

.. toctree::
   :caption: Legacy C++ interface (advanced)
   :maxdepth: 1
   :hidden:

   usage/index

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
   install/changelog.md
