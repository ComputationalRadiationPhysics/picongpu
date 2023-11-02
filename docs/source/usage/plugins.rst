.. _usage-plugins:

Plugins
=======

==================================================================================== =================================================================================
Plugin name                                                                          short description
==================================================================================== =================================================================================
:ref:`binning <usage-plugins-binningPlugin>` [#f2]_ [#f6]_                           particle binning plugin to make histograms with user-defined axes and quantity
:ref:`charge conservation <usage-plugins-chargeConservation>` [#f5]_                 maximum difference between electron charge density and div E
:ref:`checkpoint <usage-plugins-checkpoint>` [#f2]_                                  stores the primary data of the simulation for restarts.
:ref:`count particles <usage-plugins-countParticles>` [#f5]_                         count total number of macro particles
:ref:`count per supercell <usage-plugins-countPerSupercell>` [#f2]_                  count macro particles *per supercell*
:ref:`energy histogram <usage-plugins-energyHistogram>` [#f6]_                       energy histograms for electrons and ions
:ref:`energy fields <usage-plugins-energyFields>`                                    electromagnetic field energy per time step
:ref:`energy particles <usage-plugins-energyParticles>` [#f6]_                       kinetic and total energies summed over all electrons and/or ions
:ref:`ISAAC <usage-plugins-ISAAC>`                                                   interactive 3D live visualization [Matthes2016]_
:ref:`openPMD <usage-plugins-openPMD>` [#f2]_ [#f6]_                                 outputs simulation data via the openPMD API
:ref:`particle calorimeter <usage-plugins-particleCalorimeter>` [#f2]_ [#f3]_ [#f6]_ spatially resolved, particle energy detector in infinite distance
:ref:`phase space <usage-plugins-phaseSpace>` [#f2]_ [#f5]_ [#f6]_                   calculate 2D phase space [Huebl2014]_
:ref:`PNG <usage-plugins-PNG>` [#f6]_                                                pictures of 2D slices
:ref:`radiation <usage-plugins-radiation>` [#f2]_                                    compute emitted electromagnetic spectra [Pausch2012]_ [Pausch2014]_ [Pausch2018]_
:ref:`slice emittance <usage-plugins-sliceEmittance>`                                compute emittance and slice emittance of particles
:ref:`sum currents <usage-plugins-sumCurrents>` [#f5]_                               compute the total current summed over all cells
:ref:`transitionRadiation <usage-plugins-transitionRadiation>`                       compute emitted electromagnetic spectra
==================================================================================== =================================================================================

.. rubric:: Footnotes

.. [#f1] On restart, plugins with that footnote overwrite their output of previous runs.
         Manually *save* the created files of these plugins before restarting in the same directory.
.. [#f2] Requires PIConGPU to be compiled with openPMD API.
.. [#f3] Can remember particles that left the box at a certain time step.
.. [#f4] Deprecated
.. [#f5] Only runs on the *CUDA* backend (GPU).
.. [#f6] Multi-Plugin: Can be configured to run multiple times with varying parameters.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   plugins/*

Period Syntax
-------------

Most plugins allow to define a period on how often a plugin shall be executed (notified).
Its simple syntax is: ``<period>`` with a simple number.

Additionally, the following syntax allows to define intervals for periods:

``<start>:<end>[:<period>]``

* `<start>`: begin of the interval; default: 0
* `<end>`: end of the interval, including the upper bound; default: end of the simulation
* `<period>`: notify period within the interval; default: 1

Multiple intervals can be combined via a comma separated list.

Examples
^^^^^^^^

* ``42`` every 42th time step
* ``::`` equal to just writing ``1``, every time step from start (0) to the end of the simulation
* ``11:11`` only once at time step 11
* ``10:100:2`` every second time step between steps 10 and 100 (included)
* ``42,30:50:10``: at steps 30 40 42 50 84 126 168 ...
* ``5,10``: at steps 0 5 10 15 20 25 ... (only executed once per step in overlapping intervals)

Python Postprocessing
---------------------

In order to further work with the data produced by a plugin during a simulation run, PIConGPU provides python tools that can be used for reading data and visualization.
They can be found under ``lib/python/picongpu/extra/plugins``.

.. note::

   The python plugin tools have been moved to the `picongpu.extra` submodule.

It is our goal to provide at least three modules for each plugin to make postprocessing as convenient as possible:
1. a data reader (inside the ``data`` subdirectory)
2. a matplotlib visualizer (inside the ``plot_mpl`` subdirectory)
3. a jupyter widget visualizer (inside the ``jupyter_widgets`` subdirectory) for usage in jupyter-notebooks

Further information on how to use these tools can be found at each plugin page.

If you would like to help in developing those classes for a plugin of your choice, please read :ref:`python postprocessing <development-pytools>`.

.. rubric:: References

.. [Huebl2014]
        A. Huebl.
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        `DOI:10.5281/zenodo.15924 <https://doi.org/10.5281/zenodo.15924>`_

.. [Matthes2016]
        A. Matthes, A. Huebl, R. Widera, S. Grottel, S. Gumhold, and M. Bussmann
        *In situ, steerable, hardware-independent and data-structure agnostic visualization with ISAAC*,
        Supercomputing Frontiers and Innovations 3.4, pp. 30-48, (2016),
        `arXiv:1611.09048 <https://arxiv.org/abs/1611.09048>`_, `DOI:10.14529/jsfi160403 <https://doi.org/10.14529/jsfi160403>`_

.. [Huebl2017]
        A. Huebl, R. Widera, F. Schmitt, A. Matthes, N. Podhorszki, J.Y. Choi, S. Klasky, and M. Bussmann.
        *On the Scalability of Data Reduction Techniques in Current and Upcoming HPC Systems from an Application Perspective.*
        ISC High Performance Workshops 2017, LNCS 10524, pp. 15-29 (2017),
        `arXiv:1706.00522 <https://arxiv.org/abs/1706.00522>`_, `DOI:10.1007/978-3-319-67630-2_2 <https://doi.org/10.1007/978-3-319-67630-2_2>`_

.. [Pausch2012]
        R. Pausch.
        *Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics*,
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2012),
        `DOI:10.5281/zenodo.843510 <https://doi.org/10.5281/zenodo.843510>`_

.. [Pausch2014]
        R. Pausch, A. Debus, R. Widera, K. Steiniger, A.Huebl, H. Burau, M. Bussmann, and U. Schramm.
        *How to test and verify radiation diagnostics simulations within particle-in-cell frameworks*,
        Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 740, pp. 250-256 (2014)
        `DOI:10.1016/j.nima.2013.10.073 <https://doi.org/10.1016/j.nima.2013.10.073>`_

.. [Pausch2018]
        R. Pausch, A. Debus, A. Huebl, U. Schramm, K. Steiniger, R. Widera, and M. Bussmann.
        *Quantitatively consistent computation of coherent and incoherent radiation in particle-in-cell codes - a general form factor formalism for macro-particles*,
        Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 909, pp. 419-422 (2018)
        `arXiv:1802.03972 <https://arxiv.org/abs/1802.03972>`_, `DOI:10.1016/j.nima.2018.02.020 <https://doi.org/10.1016/j.nima.2018.02.020>`_

