.. _usage-plugins-phaseSpace:

Phase Space
-----------

This plugin creates a 2D phase space image for a user-given spatial and momentum coordinate.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.cfg file
^^^^^^^^^

Example for *y-pz* phase space for the *electron* species (``.cfg`` file macro):

.. code:: bash

   # Calculate a 2D phase space
   # - momentum range in m_e c
   TGB_ePSypz="--e_phaseSpace.period 10 --e_phaseSpace.space y --e_phaseSpace.momentum pz --e_phaseSpace.min -1.0 --e_phaseSpace.max 1.0"


The distinct options are (assuming a species ``e`` for electrons):

====================================== ======================================================== ============================
Option                                 Usage                                     Unit
====================================== ======================================================== ============================
``--e_phaseSpace.period <N>``          calculate each N steps                                   *none*
``--e_phaseSpace.filter``              Use filtered particles. Available filters are set up in  *none*
                                       :ref:`particleFilters.param <usage-params-core>`.
``--e_phaseSpace.space <x/y/z>``       spatial coordinate of the 2D phase space                 *none*
``--e_phaseSpace.momentum <px/py/pz>`` momentum coordinate of the 2D phase space                *none*
``--e_phaseSpace.min <ValL>``          minimum of the momentum range                            :math:`m_\mathrm{species} c`
``--e_phaseSpace.max <ValR>``          maximum of the momentum range                            :math:`m_\mathrm{species} c`
====================================== ======================================================== ============================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

locally, a counter matrix of the size local-cells of ``space`` direction times ``1024`` (for momentum bins) is permanently allocated.

Host
""""

negligible.

Output
^^^^^^

The 2D histograms are stored in ``.hdf5`` files in the ``simOutput/phaseSpace/`` directory.
A file is created per species, phasespace selection and time step.

Values are given as *charge density* per phase space bin.
In order to scale to a simpler *charge of particles* per :math:`\mathrm{d}r_i` and :math:`\mathrm{d}p_i` -bin multiply by the cell volume ``dV``.

Analysis
^^^^^^^^

The easiest way is to load the data in Python:

.. code:: python

   from picongpu.plugins.phase_space import PhaseSpace
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm
   import numpy as np


   # load data
   phase_space = PhaseSpace('/home/axel/runs/foil_001')
   e_ps, e_ps_meta = phase_space.get('e', species_filter='all', ps='ypy', iteration=1000)

   # unit conversion from SI
   mu = 1.e6  # meters to microns
   e_mc_r = 1. / (9.109e-31 * 2.9979e8)  # electrons: kg * m / s to beta * gamma

   # plotting
   plt.imshow(
       np.abs(e_ps).T * e_ps_meta.dV,
       extent = e_ps_meta.extent * [mu, mu, e_mc_r, e_mc_r],
       interpolation = 'nearest',
       aspect = 'auto',
       origin='lower',
       norm = LogNorm()
   )

   # annotations
   cbar = plt.colorbar()
   cbar.set_label(r'$Q / \mathrm{d}r \mathrm{d}p$ [$\mathrm{C s kg^{-1} m^{-2}}$]')

   ax = plt.gca()
   ax.set_xlabel(r'${0}$'.format(e_ps_meta.r) + r' [$\mathrm{\mu m}$]')
   ax.set_ylabel(r'$p_{0}$ [$\beta\gamma$]'.format(e_ps_meta.p))

   plt.show()

Note that the spatial extent of the output over time might change when running a moving window simulation.

Out-of-Range Behavior
^^^^^^^^^^^^^^^^^^^^^

Particles that are *not* in the range of ``<ValL>``/``<ValR>`` get automatically mapped to the lowest/highest bin respectively.
Take care about that when setting your range and during analysis of the results.

Known Limitations
^^^^^^^^^^^^^^^^^

- only one range per selected space-momentum-pair possible right now (naming collisions)
- charge deposition uses the counter shape for now (would need one more write to neighbours to get it correct to the shape)
- the user has to define the momentum range in advance
- the resolution is fixed to ``1024 bins`` in momentum and the number of cells in the selected spatial dimension
- this plugin does not yet use :ref:`openPMD markup <pp-openPMD>`.

References
^^^^^^^^^^

The internal algorithm is explained in `pull request #347 <https://github.com/ComputationalRadiationPhysics/picongpu/pull/347>`_ and in [Huebl2014]_.

.. [Huebl2014]
        A. Huebl.
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        chapter 3.2,
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        https://doi.org/10.5281/zenodo.15924
