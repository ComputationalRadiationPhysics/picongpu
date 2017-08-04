.. _usage-plugins-energyHistogram:

Energy Histogram
----------------

This plugin computes the energy histogram particle species simulated with PIConGPU.
The energy binning can be set up using command line parameters.
A *virtual detector* can be located in the y direction of the simulation to count only particles that would reach such a detector in an experimental setup.
If the detector setup is not specified, all particles of a species are considered in the histogram. 

.cfg files
^^^^^^^^^^

There are several command line parameters that can be used to set up this plugin.
Replace the prefix ``e`` for electrons with any other species you have defined, we keep using ``e`` in the examples below for simplicity.
Currently, the plugin can be set *once for each species*.

=========================================== ======================================================================
PIConGPU command line option                description
=========================================== ======================================================================
``--e_energyHistogram.period``              The ouput periodicity of the **electron** histogram.
                                            A value of ``100`` would mean aoutput at simulation time step *0, 100, 200, ...*.
                                            If set to a non-zero value, the energy histogram of all **electrons** is computed.
                                            By default, the value is ``0`` and no histogram for the electrons is computed.
``--e_energyHistogram.binCount``            Specifies the number of bins used for the **electron** histogram.
                                            Default is ``1024``.
``--e_energyHistogram.minEnergy``           Set the minimum energy for the **electron** histogram in *keV*.
                                            Default is ``0``, meaning *0 keV*.
``--e_energyHistogram.maxEnergy``           Set the maximum energy for the **electron** histogram in *keV*.
                                            There is **no default value**.
                                            This has to be set by the user if ``--e_energyHistogram.period 1`` is set.
``--e_energyHistogram.distanceToDetector``  Distance in *meter* of a **electron** detector located far away in y direction with slit opening in x and z direction.
                                            If set to *non-zero* value, only particles that would reach the detector are considered in the histogram.
                                            Default ``0``, meaning all particles are considered in the **electron** histogram and no detector is assumed.
``--e_energyHistogram.slitDetectorX``       Width of the **electron** detector in *meter*.
                                            If not set, all particles are counted.
``--e_energyHistogram.slitDetectorZ``       Hight of the **electron** detector in *meter*.
                                            If not set, all particles are counted.
=========================================== ======================================================================

Output
^^^^^^

The histograms are stored in ASCII files in the ``simOutput/`` directory.

The file for the electron histogram is named ``e_energyHistogram.dat`` and for all other species ``<species>_energyHistogram.dat`` likewise.
The first line of these files does not contain histogram data and is commented-out using ``#``.
It describes the energy binning that needed to interpret the following data. 
It can be seen as the head of the following data table. 
The first column is an integer value describing the simulation time step. 
The second column counts the number of real particles below the minimum energy value used for the histogram. 
The following columns give the real electron count of the particles in the specific bin described by the first line/header. 
The second last column gives the number of real particles that have a higher energy than the maximum energy used for the histogram.
The last column gives the total number of particles. 
In total there are 4 columns more than the number of bins specified with command line arguments.
Each row describes another simulation time step.

Analysis Tools
^^^^^^^^^^^^^^

The easiest way is to load the data in Python:

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # matrix with data in keV
   #   note: skips columns for iteration step, underflow bin (first 2) and
   #         overflow bin, summation of all bins (last two)
   e_ene = np.loadtxt("simOutput/e_energyHistogram.dat")[:, 2:-2]
   # upper range of each bin in keV
   #    note: only reads first row and selects the 1024 bins (default, see binCount) in columns 2:1026
   e_bins = np.loadtxt("simOutput/e_energyHistogram.dat", comments='', usecols=range(2,1026) )[0,:]
   # time steps on which histogram was written
   e_steps = np.loadtxt("simOutput/e_energyHistogram.dat", usecols=range(0,1))

   # example: plot the 10th dump (10 * .period)
   plt.plot( e_bins, e_ene[10] )
   plt.show()


Alternatively, PIConGPU comes with a command line analysis tool for the energy histograms. 
It is based on *gnuplot* and requires that gnuplot is available via command line.
The tool can be found in ``src/tools/bin/`` and is called ``BinEnergyPlot.sh``.
It accesses the gnuplot script ``BinEnergyPlot.gnuplot`` in ``src/tools/share/gnuplot/``.
``BinEnergyPlot.sh`` requires exactly three command line arguments:

======== ===================================================================
Argument Value
======== ===================================================================
1st      Path and filename to `e_energyHistogram.dat` file.
2nd      Simulation time step (needs to exist)
3rd      Label for particle count used in the graph that this tool produces.
======== ===================================================================
