.. _usage-plugins-sliceEmittance:

Slice Emittance
---------------

The plugin computes the total emittance and the slice emittance (for ten combined cells in the longitudinal direction).

Currently, it outputs only the emittance of the transverse momentum space x-px.


External Dependencies
^^^^^^^^^^^^^^^^^^^^^

None

.param file
^^^^^^^^^^^

None for now. In the future, adding more compile-time configurations might become necessary  (e.g., striding of data output). 


.cfg file
^^^^^^^^^

All options are denoted for the electron (``e``) particle species here.


================================== =========================================================================================
PIConGPU command line option       Description
================================== =========================================================================================
``--e_emittance.period arg``       compute slice emittance [for each n-th step], enable plugin by setting a non-zero value
                                   A value of ``100`` would mean an output at simulation time step *0, 100, 200, ...*.
``--e_emittance.filter arg``       Use filtered particles. All available filters will be shown with ``picongpu --help``
================================== =========================================================================================



Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

Each ``x^2``, ``p_x^2`` and ``x * p_x`` summation value as well as the number of real electrons ``gCount_e`` needs to be stored 
as ``float_64`` for each y-cell.

Host
""""

as on accelerator (needed for MPI data transfer)

Output
^^^^^^


.. note::

   This plugin is a multi plugin.
   Command line parameters can be used multiple times to create e.g. dumps with different dumping period.
   In the case where an optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   e.g.

   .. code-block:: bash

      --e_emittance.period 1000 --e_emittance.filter all
      --e_emittance.period  100 --e_emittance.filter highEnergy

   creates two plugins:

   #. slice emittance for species e each 1000th time step for **all** particles.
   #. slice emittance for species e each 100th time step **only for particles** with high energy (defined by filter).

Analysis Tools
^^^^^^^^^^^^^^

The output is a text file with the first line as a comment describing the content.
The first column is the time step.
The second column is the total emittance (of all particles defined by the filter).
Each following column is the emittance if the slice at ten cells around the position given in the comment line.


.. code:: python

   data = np.loadtxt("<path-to-emittance-file>")
   timesteps = data[:, 0]
   total_emittance = data[:, 1]
   slice_emittance = data[:, 2:]

   # time evolution of total emitance
   plt.plot(timesteps, total_emittance)
   plt.xlabel("time step")
   plt.ylabel("emittance")
   plt.show()

   # plot slice emittance over time and longitudinal (y) position
   plt.imshow(slice_emittance)
   plt.xlabel("y position [arb.u.]")
   plt.ylabel("time [arb.u.]")
   cb = plt.colorbar()
   cb.set_label("emittance")
   plt.show()

