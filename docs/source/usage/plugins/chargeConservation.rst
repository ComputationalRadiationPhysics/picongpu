.. _usage-plugins-chargeConservation:

Charge Conservation
-------------------

First the charge density of all species with respect to their shape function is computed.
Then this charge density is compared to the charge density computed from the divergence of the electric field :math:`\nabla \vec E`.
The maximum deviation value multiplied by the cell's volume is printed.

.. attention::

   This plugin assumes a Yee-like divergence E stencil!

.cfg file
^^^^^^^^^

PIConGPU command line argument (for ``.cfg`` files):

.. code:: bash

   --chargeConservation.period <periodOfSteps>

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations (needs at least one FieldTmp slot).

Host
""""

negligible.

Output and Analysis Tools
^^^^^^^^^^^^^^^^^^^^^^^^^

A new file named ``chargeConservation.dat`` is generated:

.. code::

   #timestep max-charge-deviation unit[As]
   0 7.59718e-06 5.23234e-17
   100 8.99187e-05 5.23234e-17
   200 0.000113926 5.23234e-17
   300 0.00014836 5.23234e-17
   400 0.000154502 5.23234e-17
   500 0.000164952 5.23234e-17

The charge is normalized to ``UNIT_CHARGE`` (third column) which is the typical charge of *one* macro-particle.

There is a up 5% difference to a native hdf5 post-processing based implementation of the charge conversation check due to a different order of subtraction.
And the zero-th time step (only numerical differences) might differ more then 5% relative due to the close to zero result. 



