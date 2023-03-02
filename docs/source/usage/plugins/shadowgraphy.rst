.. _usage-plugins-Shadowgraphy:

Shadowgraphy
------------

Computes a 2D image by time-integrating the Poynting-Vectors in a fixed plane in the simulation.
This can be used to extract a laser from an simulation, which obtains the full laser-plasma interactions through the PIC code.
If the probe laser propagates through plasma structures, the plasma structures lead to modulations in the probe laser's intensity, resulting in a synthetic shadowgram of the plasma structures.
The plugin performs the time-integration of the probe laser and the application of various masks on the probe pulse in Fourier space.
Thus, one needs to manually add the probe pulse to the simulation with e.g. the [incident field] param files.


External Dependencies
^^^^^^^^^^^^^^^^^^^^^
The plugin is available as soon as the :ref:`FFWT3 <install-dependencies>` is compiled in.


Usage
^^^^^
========================================= ==============================================================================================================================
Command line option                       Description
========================================= ==============================================================================================================================
``--shadowgraphy.period``                 This flag requires two equivalent integer values as in ``n:n``, ``100:100``, ``200:200``... 
                                          It describes the start time of the shadowgraphy plugin integration loop.
``--shadowgraphy.duration``               Duration of shadowgraphy calculation in simulation time steps.
                                          The plugin will be called ``duration / params::tRes`` times.
                                          The final plugin call (and the file output) happens at the time defined in ``.period`` plus the duration.
``--shadowgraphy.plane``                  Defines the plane that the slice will be parallel to. The plane is defined by its orthogonal axis. 
                                          By using 0 for the x-axis, 1 for the y-axis and 2 for the z-axis, all standard planes can be selected. 
                                          E.g. choosing the x-y-plane is done by setting the orthogonal axis to the z-axis by giving the command line argument --E_slice.plane 2.
                                          Currently only ``2`` is supported and is the default value!
``--shadowgraphy.slicePoint``             Specifies at what ratio of the total depth of the z dimension, the slice for the field extraction should be set.
                                          The value given should lie between ``0.0`` and ``1.0``.                            
``--shadowgraphy.fileName``               Output file prefix. The file is stored under fileName + "_" + (start of integration) + ":" + (end of integration) + ".dat"
``--shadowgraphy.fourieroutput``          If enabled, the fields will also be stored on disk in in ``(k_\perp, \omega)`` Fourier space.
``--shadowgraphy.intermediateoutput``     If enabled, the fields will also be stored on disk before the first transverse DFT is called.
========================================= ==============================================================================================================================

.. note::
   The current approach for the time-integration implementation with ``period`` and ``duration`` is set up this way to make it easier to implement the multi-plugin syntax later on.
   Currently the shadowgraphy plugin is not supported as a multi-plugin yet!
   It is also not possible to change the extraction plane of the fields yet, which is related to the custl-dependence of the plugin and should be implemented soon.


Output
^^^^^^
Plot the first shadowgram that is stored in the simulation output directory ``simOutput``.

.. code:: python

   import os
   import matplotlib.pyplot as plt
   import numpy as np

   def load_shadowgram(filepath):
      prevpath = os.getcwd()
      os.chdir(filepath)
      files = listdir()
      filestr = [v for v in files if v.startswith("shadowgraphy") and v.endswith(".dat")][0]
      retvals = np.loadtxt(filestr)
      os.chdir(prevpath)
      return retvals

   path = "/PATH/TO/simOutput"

   ar = load_shadowgram(path)

   fig, ax = plt.subplots(figsize=(10,10))
   ax.pcolormesh(ar)
   ax.set_aspect("equal")


The size of the cells is the size of the cells in the simulation divided by the resolution in the plugin ``CELL_WIDTH_SI / shadowgraphy::params::xRes`` and ``CELL_HEIGHT_SI / shadowgraphy::params::yRes``.
When the moving window of the simulation is activated, the resulting shadowgram is smaller in moving window propagation direction ``y`` by time it takes the laser to enter the simulation box and fully propagate through the extraction slice times the speed of light ``c``.
This prevents artifacts from the laser being cut off due to the moving window or the laser not fully being propagated through the plasma structures.


Known Issues
^^^^^^^^^^^^
* Not a multiplugin
* Dependency on custl
   * Can't change extraction plane of the plugin yet
   * Can't use multiple GPUs in the direction of the extraction plane
* The moving window can't be activated/deactivated during the plugin integration loop
   * However, the plugin works if the moving window is active or inactive from the beginning of the loop


References
^^^^^^^^^^
- *Modeling ultrafast shadowgraphy in laser-plasma interaction experiments*
   E Siminos et al 2016 Plasma Phys. Control. Fusion 58 065004
   https://doi.org/10.1088/0741-3335/58/6/065004