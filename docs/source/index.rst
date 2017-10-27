:orphan:
.. only:: html

  .. image:: ../logo/pic_logo.svg

.. only:: latex

  .. image:: ../logo/pic_logo.pdf

*A particle-in-cell code for GPGPUs*

PIConGPU is a fully relativistic, many GPGPU, 3D3V particle-in-cell (PIC) code.
The Particle-in-Cell algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of electrons and ions in the plasma based on Maxwell's equations.

How to Read This Document
-------------------------

Generally, you want to follow those pages in-order to get started.
Individual chapters are based on the information of the chapters before.

In case you are already fluent in compiling C++ projects and HPC, running PIC simulations or scientific data analysis feel free to jump the respective sections.

.. attention::
   This documentation is just getting started.
   Learn more about how to improve it :ref:`here <development-sphinx>` and please contribute via pull requests! :-)

.. note::
   We also have a `wiki`_
   and a general `official homepage`_

.. _wiki: https://github.com/ComputationalRadiationPhysics/picongpu/wiki
.. _official homepage: http://picongpu.hzdr.de

************
Installation
************
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1

   install/path
   install/instructions
   install/dependencies
   install/profile

*****
Usage
*****
.. toctree::
   :caption: USAGE
   :maxdepth: 1

   usage/reference
   usage/basics
   usage/param
   usage/particles
   usage/plugins
   usage/tbg
   usage/examples
   usage/workflows

******
Models
******
.. toctree::
   :caption: MODELS
   :maxdepth: 1

   models/pic
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

   postprocessing/python
   postprocessing/openPMD
   postprocessing/paraview

***********
Development
***********
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1

   dev/CONTRIBUTING.md
   dev/repostructure
   dev/styleguide
   dev/sphinx
   dev/doxygen
   dev/clangtools
   dev/picongpu
   dev/pmacc
   dev/doxyindex

********************
Programming Patterns
********************
.. toctree::
   :caption: PROGRAMMING PATTERNS
   :maxdepth: 1

   prgpatterns/lockstep
