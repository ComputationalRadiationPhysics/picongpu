.. _usage-plugins:

Plugins
=======

================================================================================== =======================================================================
Plugin name                                                                        short description
================================================================================== =======================================================================
:ref:`ADIOS <usage-plugins-ADIOS>` [#f2]_ [#f7]_                                   stores simulation data as openPMD flavoured ADIOS files
:ref:`energy histogram <usage-plugins-energyHistogram>` [#f7]_                     energy histograms for electrons and ions
:ref:`charge conservation <usage-plugins-chargeConservation>` [#f6]_               maximum difference between electron charge density and div E
:ref:`checkpoint <usage-plugins-checkpoint>` [#f2]_                                stores the primary data of the simulation for restarts.
:ref:`count particles <usage-plugins-countParticles>` [#f6]_                       count total number of macro particles
:ref:`count per supercell <usage-plugins-countPerSupercell>` [#f3]_                count macro particles *per supercell*
:ref:`energy fields <usage-plugins-energyFields>`                                  electromagnetic field energy per time step
:ref:`energy particles <usage-plugins-energyParticles>` [#f7]_                     kinetic and total energies summed over all electrons and/or ions
:ref:`HDF5 <usage-plugins-HDF5>` [#f2]_ [#f7]_                                     stores simulation data as openPMD flavoured HDF5 files
:ref:`ISAAC <usage-plugins-ISAAC>`                                                 interactive 3D live visualization
:ref:`intensity <usage-plugins-intensity>` [#f1]_ [#f5]_ [#f6]_                    maximum and integrated electric field along the y-direction
:ref:`particle calorimeter <usage-plugins-particleCalorimeter>` [#f3]_ [#f4]_      spatially resolved, particle energy detector in infinite distance
:ref:`particle merger <usage-plugins-particleMerger>` [#f6]_                       macro particle merging
:ref:`phase space <usage-plugins-phaseSpace>` [#f3]_ [#f6]_ [#f7]_                 calculate 2D phase space
:ref:`PNG <usage-plugins-PNG>` [#f7]_                                              pictures of 2D slices
:ref:`positions particles <usage-plugins-positionsParticles>` [#f1]_ [#f5]_ [#f6]_ save trajectory, momentum, ... of a *single* particle
:ref:`radiation <usage-plugins-radiation>` [#f3]_                                  compute emitted electromagnetic spectra
:ref:`resource log <usage-plugins-resourceLog>`                                    monitor used hardware resources & memory
:ref:`slice field printer <usage-plugins-sliceFieldPrinter>` [#f5]_                print out a slice of the electric and/or magnetic and/or current field
:ref:`sum currents <usage-plugins-sumCurrents>`                                    compute the total current summed over all cells
================================================================================== =======================================================================

.. rubric:: Footnotes

.. [#f1] On restart, plugins with that footnote overwrite their output of previous runs.
         Manually *save* the created files of these plugins before restarting in the same directory.
.. [#f2] Either *ADIOS* or *HDF5* is required for simulation restarts.
         If both are available, writing checkpoints with ADIOS is automatically preferred by the simulation.
.. [#f3] Requires *HDF5* for output.
.. [#f4] Can remember particles that left the box at a certain time step.
.. [#f5] Deprecated
.. [#f6] Only runs on the *CUDA* backend (GPU).
.. [#f7] Multi-Plugin: Can be configured to run multiple times with varying parameters.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   plugins/*
