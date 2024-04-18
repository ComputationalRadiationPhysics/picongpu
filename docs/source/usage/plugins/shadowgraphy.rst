.. _usage-plugins-Shadowgraphy:

Shadowgraphy
------------

Computes a 2D image by time-integrating the Poynting-Vectors in a fixed plane in the simulation.
This can be used to extract a laser from an simulation, which obtains the full laser-plasma interactions through the PIC code.
If the probe laser propagates through plasma structures, the plasma structures lead to modulations in the probe laser's intensity, resulting in a synthetic shadowgram of the plasma structures.
The plugin performs the time-integration of the probe laser and the application of various masks on the probe pulse in Fourier space.
Thus, one needs to manually add the probe pulse to the simulation with e.g. the :ref:`incident field <usage/param/core:incidentField.param>` param files.
Since the plugin currently only works in the xy-plane, the probe pulse should propagate in z direction.
The integration plane for the plugin must lie within the simulation volume and the position of the field absorber should be considered when placing the plugin plane.


External Dependencies
^^^^^^^^^^^^^^^^^^^^^
The plugin is available as soon as the :ref:`FFWT3 <install-dependencies>` is compiled in.


Usage
^^^^^
========================================= ==============================================================================================================================
Command line option                       Description
========================================= ==============================================================================================================================
``--shadowgraphy.start``                  Step to start the plugin and with this the time integration.
``--shadowgraphy.duration``               Duration of shadowgraphy calculation in simulation time steps.
                                          The plugin will be called ``duration / params::tRes`` times.
                                          The final plugin call (and the file output) happens at the time defined in ``.period`` plus the duration.
``--shadowgraphy.slicePoint``             Specifies at what ratio of the total depth of the z dimension, the slice for the field extraction should be set.
                                          The value given should lie between ``0.0`` and ``1.0``.                            
``--shadowgraphy.focusPos``               Focus position of lens system relative to slicePoint. The focus position is given in SI units. 
``--shadowgraphy.fileName``               Output file prefix for openPMD output.          
``--shadowgraphy.ext``                    Backend for openPMD output.
``--shadowgraphy.fourierOutput``          If enabled, the fields will also be stored on disk in in ``(x, y, \omega)`` Fourier space in an openPMD file.
========================================= ==============================================================================================================================

.. note::
   Currently the plugin only supports an integration slice in the xy-plane, which means that for probing setups the probe pulse should propagate in z direction.
   The moving window can't be activated or deactivated during the plugin integration loop.

========================================= ==============================================================================================================================
Reqired param file options                Description
========================================= ==============================================================================================================================
``tRes``                                  Use the fields at each tRes'th time-step of simulation
``xRes``                                  Use each xRes'th field value in x direction
``yRes``                                  Use each yRes'th field value in y direction
``omegaWfMin``                            Minimum non-zero value for abs(omega)
``omegaWfMax``                            Maximum non-zero value for abs(omega)
``masks::positionWf``                     Mask that is multiplied to E and B field when gathering the slice
``masks::timeWf``                         Mask that is multiplied to E and B field during the time integration
``masks::maskFourier``                    Mask that is multiplied to E and B field in Fourier domain
========================================= ==============================================================================================================================


Output
^^^^^^
Plot the first shadowgram that is stored in the simulation output directory ``simOutput``.

.. code:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import openpmd_api as io


   def load_shadowgram(series):
      i = series.iterations[[i for i in series.iterations][0]]

      shadowgram_tmp = i.meshes["shadowgram"][io.Mesh_Record_Component.SCALAR].load_chunk()
      unit = i.meshes["shadowgram"].get_attribute("unitSI")
      series.flush()

      return shadowgram_tmp * unit


   def load_meshgrids(series):
      i = series.iterations[[i for i in series.iterations][0]]

      xspace_tmp = i.meshes["Spatial positions"]["x"].load_chunk()
      xunit = i.meshes["Spatial positions"]["x"].get_attribute("unitSI")
      series.flush()
      xspace = xspace_tmp * xunit

      yspace_tmp = i.meshes["Spatial positions"]["y"].load_chunk()
      yunit = i.meshes["Spatial positions"]["y"].get_attribute("unitSI")
      series.flush()
      yspace = yspace_tmp * yunit

      return np.meshgrid(xspace, yspace)


   path = "PATH/TO/simOutput"

   series = io.Series(path + "/shadowgraphy_" + "%T." + "bp5", io.Access.read_only)
   shadowgram = load_shadowgram(series)
   xm, ym = load_meshgrids(series)
   series.close()

   fig, ax = plt.subplots(figsize=(10, 10))
   ax.pcolormesh(xm, ym, shadowgram)
   ax.set_aspect("equal")


Shadowgram Size and Moving Window
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The size of the pixels is the size of the cells in the simulation divided by the resolution in the plugin ``CELL_WIDTH_SI * shadowgraphy::params::xRes`` and ``CELL_HEIGHT_SI * shadowgraphy::params::yRes``.
The shadowgram itself does not include cells that lie outside of the field absorber in both x and y direction.
When the moving window is activated, the resulting shadowgram is smaller in moving window propagation direction ``y``. 
The size difference is equal to the speed of light times the time it would take for light to propagate from the ``-z`` border of the simulation box to the plugin integration plane plus the integration duration.
This prevents artifacts from the laser being cut off due to the moving window or the laser not fully being propagated through the plasma structures.


References
^^^^^^^^^^
- *Modeling ultrafast shadowgraphy in laser-plasma interaction experiments*
   E Siminos et al 2016 Plasma Phys. Control. Fusion 58 065004
   https://doi.org/10.1088/0741-3335/58/6/065004
- *Synthetic few-cycle shadowgraphy diagnostics in particle-in-cell codes for characterizing laser-plasma accelerators*
   Carstens, F.-O.,
   Master Thesis on shadowgraphy plugin
   https://doi.org/10.5281/zenodo.7755263
