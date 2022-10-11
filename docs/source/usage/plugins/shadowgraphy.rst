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


Output
^^^^^^


Known Issues
^^^^^^^^^^^^


References
^^^^^^^^^^

- *Modeling ultrafast shadowgraphy in laser-plasma interaction experiments*
   E Siminos et al 2016 Plasma Phys. Control. Fusion 58 065004
   https://doi.org/10.1088/0741-3335/58/6/065004