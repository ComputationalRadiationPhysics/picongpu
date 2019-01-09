.. _usage-plugins-energyHistogram:

Energy Histogram
----------------

This plugin computes the energy histogram (spectrum) of a selected particle species and stores it to plain text files.
The acceptance of particles for counting in the energy histogram can be adjusted, e.g. to model the limited acceptance of a realistic spectrometer.

.param file
^^^^^^^^^^^

The :ref:`particleFilters.param <usage-params-core>` file allows to define accepted particles for the energy histogram.
A typical :ref:`filter <usage-params-core-particles-filters>` could select particles within a specified :ref:`opening angle in forward direction <usage-workflows-particleFilters>`.

.cfg files
^^^^^^^^^^

There are several command line parameters that can be used to set up this plugin.
Replace the prefix ``e`` for electrons with any other species you have defined, we keep using ``e`` in the examples below for simplicity.
Currently, the plugin can be set *once for each species*.

=========================================== =====================================================================================
PIConGPU command line option                description
=========================================== =====================================================================================
``--e_energyHistogram.period``              The output periodicity of the **electron** histogram.
                                            A value of ``100`` would mean an output at simulation time step *0, 100, 200, ...*.
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
=========================================== =====================================================================================

.. note::

   This plugin is a multi plugin.
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   In the case where an optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   For example,

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

Data Reader
"""""""""""
You can quickly load and interact with the data in Python with:

.. code:: python

   from picongpu.plugins.data import EnergyHistogramData

   eh_data = EnergyHistogramData('/home/axel/runs/lwfa_001')

   # show available iterations
   eh_data.get_iterations(species='e')

   # show available simulation times
   eh_data.get_times(species='e')

   # load data for a given iteration
   counts, bins_keV = eh_data.get('e', species_filter='all', iteration=2000)

   # load data for a given time
   counts, bins_keV = eh_data.get('e', species_filter='all', time=1.3900e-14)

   # get data for multiple iterations
   d, bins, iteration, dt = eh_data.get(species='e', iteration=[200, 400, 8000])


Matplotlib Visualizer
"""""""""""""""""""""

You can quickly plot the data in Python with:

.. code:: python

   from picongpu.plugins.plot_mpl import EnergyHistogramMPL
   import matplotlib.pyplot as plt


   # create a figure and axes
   fig, ax = plt.subplots(1, 1)

   # create the visualizer
   eh_vis = EnergyHistogramMPL('path/to/run_dir', ax)

   eh_vis.visualize(iteration=200, species='e')

   plt.show()

   # specifying simulation time is also possible (granted there is a matching iteration for that time)
   eh_vis.visualize(time=2.6410e-13, species='e')

   plt.show()

   # plotting histogram data for multiple simulations simultaneously also works:
   eh_vis = EnergyHistogramMPL([
        ("sim1", "path/to/sim1"),
        ("sim2", "path/to/sim2"),
        ("sim3", "path/to/sim3")], ax)
    eh_vis.visualize(species="e", iteration=10000)

    plt.show()


The visualizer can also be used from the command line (for a single simulation only) by writing

 .. code:: bash

    python energy_histogram_visualizer.py

with the following command line options

================================     ======================================================
Options                              Value
================================     ======================================================
-p                                   Path to the run directory of a simulation.
-i                                   An iteration number
-s (optional, defaults to 'e')       Particle species abbreviation (e.g. 'e' for electrons)
-f (optional, defaults to 'all')     Species filter string
================================     ======================================================



Alternatively, PIConGPU comes with a command line analysis tool for the energy histograms.
It is based on *gnuplot* and requires that gnuplot is available via command line.
The tool can be found in ``src/tools/bin/`` and is called ``BinEnergyPlot.sh``.
It accesses the gnuplot script ``BinEnergyPlot.gnuplot`` in ``src/tools/share/gnuplot/``.
``BinEnergyPlot.sh`` requires exactly three command line arguments:

======== ===================================================================
Argument Value
======== ===================================================================
1st      Path and filename to ``e_energyHistogram.dat`` file.
2nd      Simulation time step (needs to exist)
3rd      Label for particle count used in the graph that this tool produces.
======== ===================================================================



Jupyter Widget
""""""""""""""

If you want more interactive visualization, then start a jupyter notebook and make
sure that ``ipywidgets`` and ``ìpympl`` are installed.

After starting the notebook server write the following

.. code:: python

   # this is required!
   %matplotlib widget
   import matplotlib.pyplot as plt
   plt.ioff()

   from IPython.display import display
   from picongpu.plugins.jupyter_widgets import EnergyHistogramWidget

   # provide the paths to the simulations you want to be able to choose from
   # together with labels that will be used in the plot legends so you still know
   # which data belongs to which simulation
   w = EnergyHistogramWidget(run_dir_options=[
           ("scan1/sim4", scan1_sim4),
           ("scan1/sim5", scan1_sim5)])
   display(w)

and then interact with the displayed widgets.
