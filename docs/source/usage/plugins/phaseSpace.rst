.. _usage-plugins-phaseSpace:

Phase Space
-----------

This plugin creates a 2D phase space image for a user-given spatial and momentum coordinate.

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

The output per bin is given in *charge density*.
To get a *charge of particles per dr_i and dp_i bin* multiply by:

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

To retrieve the offset of the spatial dimension due to a moving window additional attributes are provided.

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

References
^^^^^^^^^^

The internal algorithm is explained in `pull request #347 <https://github.com/ComputationalRadiationPhysics/picongpu/pull/347>`_ .
