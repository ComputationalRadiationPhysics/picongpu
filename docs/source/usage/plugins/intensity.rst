.. _usage-plugins-intensity:

Intensity
---------

The maximum amplitude of the electric field for each cell in y-cell-position in **V/m** and the integrated amplitude of the electric field (integrated over the entirer x- and z-extent of the simulated volume and given for each y-cell-position).


.. attention::

   There might be an error in the units of the integrated output.

.. note::

   A renaming of this plugin would be very useful in order to understand its purpose more intuitively. 

.cfg file
^^^^^^^^^

By setting the PIConGPU command line flag ``--intensity.period`` to a non-zero value the plugin computes the maximum electric field and the integrated electric field for each cell-wide slice in y-direction. 
The default value is ``0``, meaning that nothing is computed.
By setting e.g. ``--intensity.period 100`` the electric field analysis is computed for time steps *0, 100, 200, ...*.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

negligible.

Host
""""

negligible.

Output
^^^^^^

The output of the maximum electric field for each y-slice is stored in ``Intensity_max.dat``.
The output of the integrated electric field for each y-slice is stored in ``Intensity_integrated.dat``.

Both files have two header rows describing the data.
.. code::

   #step position_in_laser_propagation_direction
   #step amplitude_data[*]

The following odd rows give the time step and then describe the y-position of the slice at which the maximum electric field or integrated electric field is computed.
The even rows give the time step again and then the data (maximum electric field or integrated electric field) at the positions given in the previews row.
 
Know Issues
^^^^^^^^^^^

Currently, the output file is overwritten after restart.
Additionally, this plugin does not work with non-regular domains, see `here <https://github.com/ComputationalRadiationPhysics/picongpu/blob/4a6d8ed0ea4a1bf54f55b4941461c6368df89b1c/src/picongpu/include/plugins/IntensityPlugin.hpp#L235>`_ .
This will be fixed in a future version. 

There might be an error in the units of the integrated output.

For a full list, see `#327 <https://github.com/ComputationalRadiationPhysics/picongpu/issues/327>`_ .

