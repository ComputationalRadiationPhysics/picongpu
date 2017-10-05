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

====================================== ========================================= ============================
Option                                 Usage                                     Unit
====================================== ========================================= ============================
``--e_phaseSpace.period <N>``          calculate each N steps                    *none*
``--e_phaseSpace.space <x/y/z>``       spatial coordinate of the 2D phase space  *none*
``--e_phaseSpace.momentum <px/py/pz>`` momentum coordinate of the 2D phase space *none*
``--e_phaseSpace.min <ValL>``          minimum of the momentum range             :math:`m_\mathrm{species} c`
``--e_phaseSpace.max <ValR>``          maximum of the momentum range             :math:`m_\mathrm{species} c`
====================================== ========================================= ============================

Output
^^^^^^

Values are given as *charge density* per phase space bin.
In order to scale to a simpler *charge of particles* per :math:`\mathrm{d}r_i` and :math:`\mathrm{d}p_i` -bin multiply by the cell volume:

.. code:: python

   ps = ... # get data set from h5py

   # 3D3V example
   dV = ps.attrs['dV'] * ps.attrs['dr_unit']**3
   charge_per_bin = ps[:,:] * ps.attrs['sim_unit'] * dV

Out-of-Range Behavior
^^^^^^^^^^^^^^^^^^^^^

Particles that are *not* in the range of ``<ValL>``/``<ValR>`` get automatically mapped to the lowest/highest bin respectively.
Take care about that when setting your range and during analysis of the results.

Spatial Offset
^^^^^^^^^^^^^^

The spatial extent of the output might change due to a moving window.
Additional attributes are provided to retrieve that spatial information:

See this python example:

.. code:: python

   ps = ... # get data set from h5py

   mv_start = ps.attrs['movingWindowOffset']
   mv_end = mv_start + ps.attrs['movingWindowSize']
   spatial_offset = ps.attrs['_global_start'][1] # 2D data set: 0 (r_i); 1 (p_i)

   dr = ps.attrs['dr'] * ps.attrs['dr_unit']

   spatial_extend_cells = np.array([mv_start, mv_end]) + spatial_offset
   spatial_extend = spatial_extend_cells * dr

   # Cut out the current window
   ps_cut = ps[mv_start:mv_end, :]


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
