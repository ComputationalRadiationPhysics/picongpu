:orphan:
.. only:: html

  .. image:: ../logo/pic_logo.svg

.. only:: latex

  .. image:: ../logo/pic_logo.pdf

*Particle-in-Cell Simulations for the Exascale Era*

PIConGPU is a fully relativistic, manycore, 3D3V and 2D3V particle-in-cell (PIC) code.
The PIC algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of electrons and ions in the plasma based on the Vlasov-Maxwell system of equations.

How to Read This Document
-------------------------

Generally, to get started **follow the manual pages in order**.
Individual chapters are based on the information in the chapters before.
In case you are already fluent in compiling C++ projects and HPC, running PIC simulations or scientific data analysis, feel free to jump the respective sections.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of PIConGPU.
   If you are looking for the latest *development* version, `click here <https://picongpu.readthedocs.io/en/latest/>`_.


.. note::

   We are migrating our `wiki`_ to this manual, but some pages might still be missing.
   We also have an `official homepage`_ .

.. note::

   Are you looking for our latest Doxygen docs for the API?

   See http://computationalradiationphysics.github.io/picongpu

.. _wiki: https://github.com/ComputationalRadiationPhysics/picongpu/wiki
.. _official homepage: http://picongpu.hzdr.de

************
Installation
************
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/path
   install/instructions
   install/dependencies
   install/profile
   install/changelog.md

*****
Usage
*****
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/reference
   usage/basics
   usage/param
   usage/plugins
   usage/tbg
   usage/python
   usage/examples
   usage/workflows

******
Models
******
.. toctree::
   :caption: MODELS
   :maxdepth: 1
   :hidden:

   models/pic
   models/AOFDTD
   models/shapes
   models/LL_RR
   models/field_ionization
   models/collisional_ionization
   models/photons

***************
Post-Processing
***************
.. toctree::
   :caption: Post-Processing
   :maxdepth: 2
   :hidden:

   postprocessing/python
   postprocessing/openPMD
   postprocessing/paraview

***********
Development
***********
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   dev/CONTRIBUTING.md
   dev/docs/COMMIT.md
   dev/repostructure
   dev/styleguide
   dev/sphinx
   dev/doxygen
   dev/clangtools
   dev/picongpu
   dev/pmacc
   dev/py_postprocessing
   dev/doxyindex

********************
Programming Patterns
********************
.. toctree::
   :caption: PROGRAMMING PATTERNS
   :maxdepth: 1
   :hidden:

   prgpatterns/lockstep
