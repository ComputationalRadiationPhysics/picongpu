.. _usage-plugins-saxs:

SAXS
---------

This plugin calculates Small Angle X-ray Scattering (SAXS) patterns from particle positions.

.. math::

   I({\bf q}) &= \frac{F({\bf q})F({\bf q})^*}{N} \\
   F({\bf q}) &= \sum^{N}_{j=1}f_j\exp({\bf q \cdot r}_j)

============================== ================================================================================
Variable                       Meaning
============================== ================================================================================
:math:`\bf r_j`                The position of particle *j* .
:math:`\bf q`                  Scattering vector, which is defined as :math:`|{\bf q}| = \frac{4 \pi \sin \theta}{\lambda}`.
:math:`f_j`                    The form factor of particle *j*.
============================== ================================================================================

This is based on the kinematic model of scattering. It can calculate scattering intensity from all the particle positions for each time step. Right now, only the plane wave condition is considered. 
Multiple Scattering CAN NOT be handled in this model.

Temporal and transversal x-ray envelope will be take into account in the future.

For free electrons, :math:`f_j = 1`. For ions, further computations are required, which is not implemented yet. 

.cfg file
^^^^^^^^^

For a specific (charged) species ``<species>`` e.g. ``e``, the scattering can be computed by the following commands.  

========================================= ==============================================================================================================================
Command line option                       Description
========================================= ==============================================================================================================================
``--<species>_saxs.period``               Gives the number of time steps between which the scattering intensity should be calculated.
                                          Default is ``0``, which means that the scattering intensity in never calculated and therefor off.
                                          Using ``1`` calculates the scattering intensity constantly. Any value ``>=2`` is currently producing nonsense.
``--<species>.qx_max``                    Upper bound of reciprocal space range in qx direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.qy_max``                    Upper bound of reciprocal space range in qy direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.qz_max``                    Upper bound of reciprocal space range in qz direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.qx_min``                    Lower bound of reciprocal space range in qx direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.qy_min``                    Lower bound of reciprocal space range in qy direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.qz_min``                    Lower bound of reciprocal space range in qz direction. The unit is :math:`Å^{-1}`
                                          Default is ``5``.
``--<species>.n_qx``                      Number of scattering vectors needed to be calculated in qx direction. The unit is :math:`Å^{-1}`
                                          Default is ``100``.
``--<species>.n_qy``                      Number of scattering vectors needed to be calculated in qy direction. The unit is :math:`Å^{-1}`
                                          Default is ``100``.
``--<species>.n_qz``                      Number of scattering vectors needed to be calculated in qz direction. The unit is :math:`Å^{-1}`
                                          Default is ``1``.
========================================= ==============================================================================================================================


Output
^^^^^^

``<species>_saxs.[timestep].dat``

An *ASCII* file that contains scattering intensity for each scattering vector defined by the reciprocal space range. The first 3 columns are the 3 components of scattering vector, and the 4th column is the corresponding scattering intensity.

``<species>_saxs.[timestep].log``

An *ASCII* file that contains the number of real particles and the number of macro particles. 


References
^^^^^^^^^^

- *GAPD* : a GPU-accelerated atom-based polychromatic diffraction simulation code <https://doi.org/10.1107/S1600577517016733>,
  a Journal of Synchrotron Radiation paper of the prototype algorithm.