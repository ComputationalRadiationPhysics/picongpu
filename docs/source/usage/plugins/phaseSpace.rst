.. _usage-plugins-phaseSpace:

Phase Space
-----------

This plugin creates a 2D phase space image for a user-given spatial and momentum coordinate.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.cfg file
^^^^^^^^^

Example for *y-pz* phase space for the *electron* species (``.cfg`` file macro):

.. code:: bash

   # Calculate a 2D phase space
   # - momentum range in m_e c
   TGB_ePSypz="--e_phaseSpace.period 10 --e_phaseSpace.filter all --e_phaseSpace.space y --e_phaseSpace.momentum pz --e_phaseSpace.min -1.0 --e_phaseSpace.max 1.0"


The distinct options are (assuming a species ``e`` for electrons):

====================================== ======================================================== ============================
Option                                 Usage                                     Unit
====================================== ======================================================== ============================
``--e_phaseSpace.period <N>``          calculate each N steps                                   *none*
``--e_phaseSpace.filter``              Use filtered particles. Available filters are set up in  *none*
                                       :ref:`particleFilters.param <usage-params-core>`.
``--e_phaseSpace.space <x/y/z>``       spatial coordinate of the 2D phase space                 *none*
``--e_phaseSpace.momentum <px/py/pz>`` momentum coordinate of the 2D phase space                *none*
``--e_phaseSpace.min <ValL>``          minimum of the momentum range                            :math:`m_\mathrm{species} c`
``--e_phaseSpace.max <ValR>``          maximum of the momentum range                            :math:`m_\mathrm{species} c`
====================================== ======================================================== ============================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

locally, a counter matrix of the size local-cells of ``space`` direction times ``1024`` (for momentum bins) is permanently allocated.

Host
""""

negligible.

Output
^^^^^^

The 2D histograms are stored in ``.hdf5`` files in the ``simOutput/phaseSpace/`` directory.
A file is created per species, phasespace selection and time step.

Values are given as *charge density* per phase space bin.
In order to scale to a simpler *charge of particles* per :math:`\mathrm{d}r_i` and :math:`\mathrm{d}p_i` -bin multiply by the cell volume ``dV``.

Analysis Tools
^^^^^^^^^^^^^^

Data Reader
"""""""""""
You can quickly load and interact with the data in Python with:

.. code:: python

   from picongpu.plugins.data import PhaseSpaceData
   import numpy as np


   ps_data = PhaseSpaceData('/home/axel/runs/lwfa_001')
   # show available iterations
   ps_data.get_iterations(ps="xpx", species="e", species_filter='all')

   # show available simulation times
   ps_data.get_times(ps="xpx", species="e", species_filter='all')

   # load data for a given iteration
   ps, meta = ps_data.get(ps='ypy', species='e', species_filter='all', iteration=2000)

   # unit conversion from SI
   mu = 1.e6  # meters to microns
   e_mc_r = 1. / (9.109e-31 * 2.9979e8)  # electrons: kg * m / s to beta * gamma

   Q_dr_dp = np.abs(ps) * meta.dV  # C s kg^-1 m^-2
   extent = meta.extent * [mu, mu, e_mc_r, e_mc_r]  # spatial: microns, momentum: beta*gamma

   # load data for a given time
   ps, ps_meta = ps_data.get(ps="xpx", species="e", species_filter='all', time=1.3900e-14)

   # load data for multiple iterations
   ret = ps_data.get(ps="xpx", species="e", species_filter='all', iteration=[2000, 4000])

   # data and metadata for iteration 2000
   # (data is in same order as the value passed to the 'iteration' parameter)
   ps, meta = ret[0]


Note that the spatial extent of the output over time might change when running a moving window simulation.

Matplotlib Visualizer
"""""""""""""""""""""

You can quickly plot the data in Python with:

.. code:: python

   from picongpu.plugins.plot_mpl import PhaseSpaceMPL
   import matplotlib.pyplot as plt


   # create a figure and axes
   fig, ax = plt.subplots(1, 1)

   # create the visualizer
   ps_vis = PhaseSpaceMPL('path/to/run_dir', ax)

   # plot
   ps_vis.visualize(iteration=200, species='e')

   plt.show()

   # specifying simulation time is also possible (granted there is a matching iteration for that time)
   ps_vis.visualize(time=2.6410e-13, species='e')

   plt.show()

   # plotting data for multiple simulations simultaneously also works:
   ps_vis = PhaseSpaceMPL([
        ("sim1", "path/to/sim1"),
        ("sim2", "path/to/sim2"),
        ("sim3", "path/to/sim3")], ax)
    ps_vis.visualize(species="e", iteration=10000)

    plt.show()


The visualizer can also be used from the command line (for a single simulation only) by writing

 .. code:: bash

    python phase_space_visualizer.py

with the following command line options

================================     =======================================================
Options                              Value
================================     =======================================================
-p                                   Path and filename to the run directory of a simulation.
-i                                   An iteration number
-s (optional, defaults to 'e')       Particle species abbreviation (e.g. 'e' for electrons)
-f (optional, defaults to 'all')     Species filter string
-m (optional, defaults to 'ypy')     Momentum string to specify the phase space
================================     =======================================================

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
   from picongpu.plugins.jupyter_widgets import PhaseSpaceWidget

   # provide the paths to the simulations you want to be able to choose from
   # together with labels that will be used in the plot legends so you still know
   # which data belongs to which simulation
   w = PhaseSpaceWidget(run_dir_options=[
           ("scan1/sim4", scan1_sim4),
           ("scan1/sim5", scan1_sim5)])
   display(w)


and then interact with the displayed widgets.


Out-of-Range Behavior
^^^^^^^^^^^^^^^^^^^^^

Particles that are *not* in the range of ``<ValL>``/``<ValR>`` get automatically mapped to the lowest/highest bin respectively.
Take care about that when setting your range and during analysis of the results.

Known Limitations
^^^^^^^^^^^^^^^^^

- only one range per selected space-momentum-pair possible right now (naming collisions)
- charge deposition uses the counter shape for now (would need one more write to neighbors to evaluate it correctly according to the shape)
- the user has to define the momentum range in advance
- the resolution is fixed to ``1024 bins`` in momentum and the number of cells in the selected spatial dimension
- this plugin does not yet use :ref:`openPMD markup <pp-openPMD>`.

References
^^^^^^^^^^

The internal algorithm is explained in `pull request #347 <https://github.com/ComputationalRadiationPhysics/picongpu/pull/347>`_ and in [Huebl2014]_.

.. [Huebl2014]
        A. Huebl.
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        chapter 3.2,
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        https://doi.org/10.5281/zenodo.15924
