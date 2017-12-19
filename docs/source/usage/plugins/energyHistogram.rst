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

=========================================== =====================================================================================
PIConGPU command line option                description
=========================================== =====================================================================================
``--e_energyHistogram.period``              The ouput periodicity of the **electron** histogram.
                                            A value of ``100`` would mean aoutput at simulation time step *0, 100, 200, ...*.
                                            If set to a non-zero value, the energy histogram of all **electrons** is computed.
                                            By default, the value is ``0`` and no histogram for the electrons is computed.
``--e_energy.filter``                       Use filtered particles. Available filters are set up in 
                                            :ref:`particleFilters.param <usage-params-core>`.
``--e_energyHistogram.binCount``            Specifies the number of bins used for the **electron** histogram.
                                            Default is ``1024``.
``--e_energyHistogram.minEnergy``           Set the minimum energy for the **electron** histogram in *keV*.
                                            Default is ``0``, meaning *0 keV*.
``--e_energyHistogram.maxEnergy``           Set the maximum energy for the **electron** histogram in *keV*.
                                            There is **no default value**.
                                            This has to be set by the user if ``--e_energyHistogram.period 1`` is set.
``--e_energyHistogram.distanceToDetector``  Distance in *meter* of a **electron** detector located far away in y direction with
                                            slit opening in x and z direction. If set to *non-zero* value, only particles that 
                                            would reach the detector are considered in the histogram.
                                            Default ``0``, meaning all particles are considered in the **electron** histogram 
                                            and no detector is assumed.
``--e_energyHistogram.slitDetectorX``       Width of the **electron** detector in *meter*.
                                            If not set, all particles are counted.
``--e_energyHistogram.slitDetectorZ``       Hight of the **electron** detector in *meter*.
                                            If not set, all particles are counted.
=========================================== =====================================================================================

.. note::

   This plugin is a multi plugin. 
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   In the case where an optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   e.g. 

   .. code-block:: bash

      --e_energyHistogram.period 128 --e_energyHistogram.filter all --e_energyHistogram.maxEnergy 10
      --e_energyHistogram.period 100 --e_energyHistogram.filter all --e_energyHistogram.maxEnergy 20 --e_energyHistogram.binCount 512

   creates two plugins:
 
   #. create an electron histogram **with 512 bins** each 128th time step.
   #. create an electron histogram **with 1024 bins** (this is the default) each 100th time step.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

an extra array with the number of bins.

Host
""""

negligible.

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

You can quickly plot the data in Python with:

.. code:: python

   from picongpu.plugins.energy_histogram import EnergyHistogram
   import matplotlib.pyplot as plt

   # load data
   energy_histogram = EnergyHistogram('/home/axel/runs/lwfa_001')
   counts, bins = energy_histogram.get('e', species_filter='all', iteration=2000)

   # unit conversion
   MeV = 1.e-3  # keV to MeV

   # plotting
   plt.plot(bins * MeV, counts)

   # range
   ax = plt.gca()
   # log scale example
   # ax.set_yscale('log')
   # ax.set_ylim([1.e7, 1.e12])

   # annotations
   ax.set_xlabel(r'E$_\mathrm{kin}$ [MeV]')
   ax.set_ylabel(r'count [arb.u.]')

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
