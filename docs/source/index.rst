.. image:: ../logo/pic_logo.svg

A particle-in-cell code for GPGPUs

Introduction
============

PIConGPU is a fully relativistic, many GPGPU, 3D3V particle-in-cell (PIC) code.
The Particle-in-Cell algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of electrons and ions in the plasma based on Maxwell's equations.

.. attention::
   This documentation is just getting started.
   Learn more about how to improve it :ref:`here <development-sphinx>` and please contribute via pull requests! :-)

.. note::
   We also have a `wiki`_
   and a general `official homepage`_

.. _wiki: https://github.com/ComputationalRadiationPhysics/picongpu/wiki
.. _official homepage: http://picongpu.hzdr.de

.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1

   install/path
   install/INSTALL.md
   install/profile

.. toctree::
   :caption: USAGE
   :maxdepth: 1

   usage/reference
   usage/basics
   usage/param
   usage/particles
   usage/plugin
   usage/tbg
   usage/examples

.. toctree::
   :caption: MODELS
   :maxdepth: 1

   models/pic
   models/LL_RR
   models/ionization
   models/photons

.. toctree::
   :caption: Post-Processing
   :maxdepth: 2

   postprocessing/python
   postprocessing/openPMD
   postprocessing/paraview

.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1

   dev/CONTRIBUTING.md
   dev/sphinx
   dev/picongpu
   dev/pmacc
   dev/doxyindex
