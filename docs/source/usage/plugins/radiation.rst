.. _usage-plugins-radiation:

Radiation
---------

The spectrally resolved far field radiation of charged macro particles.

Our simulation computes the `Lienard Wiechert potentials <https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential>`_ to calculate the emitted electromagnetic spectra for different observation directions using the far field approximation.

.. math::

   \frac{\operatorname{d}^2I}{\operatorname{d}{\Omega}\operatorname{d}\omega}\left(\omega,\vec{n}\right)= \frac{q^2}{16\pi^3\varepsilon_0 c} \left|\sum\limits_{k=1}^{N}\int\limits_{-\infty}^{+\infty}\frac{\vec{n}\times\left[\left(\vec{n}-\vec{\beta}_k(t)\right)\times\dot{\vec{\beta}}_k(t)\right]}{\left(1-\vec{\beta}_k(t)\cdot\vec{n}\right)^2}\cdot\operatorname{e}^{\operatorname{i}\omega\left(t-\vec{n}\cdot\vec{r}_k(t)/c\right)}\operatorname{d}t\right|^2

Details on how radiation is computed with this plugin and how the plugin works can be found in [Pausch2012]_.
A list of tests can be found in [Pausch2014]_ and [Pausch2019]_.

============================== ================================================================================
Variable                       Meaning
============================== ================================================================================
:math:`\vec r_k(t)`            The position of particle *k* at time *t*.
:math:`\vec \beta_k(t)`        The normalized speed of particle *k* at time *t*.
                               (Speed divided by the speed of light)
:math:`\dot{\vec{\beta}}_k(t)` The normalized acceleration of particle *k* at time *t*.
                               (Time derivative of the normalized speed.)
:math:`t`                      Time
:math:`\vec n`                 Unit vector pointing in the direction where the far field radiation is observed.
:math:`\omega`                  The circular frequency of the radiation that is observed.
:math:`N`                      Number of all (macro) particles that are used for computing the radiation.
:math:`k`                      Running index of the particles.
============================== ================================================================================

Currently this allows to predict the emitted radiation from plasma if it can be described by classical means.
Not considered are emissions from ionization, Compton scattering or any bremsstrahlung that originate from scattering on scales smaller than the PIC cell size. 

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.param files
^^^^^^^^^^^^

In order to setup the radiation analyzer plugin, both the :ref:`radiation.param <usage-params-plugins>` and the :ref:`radiationObserver.param <usage-params-plugins>` have to be configured **and** the radiating particles need to have the attribute ``momentumPrev1`` which can be added in :ref:`speciesDefinition.param <usage-params-core>`.

In *radiation.param*, the number of frequencies ``N_omega`` and observation directions ``N_theta`` is defined.

Frequency range
"""""""""""""""

The frequency range is set up by choosing a specific namespace that defines the frequency setup

.. code:: cpp

   /* choose linear frequency range */
   namespace radiation_frequencies = linear_frequencies;

Currently you can choose from the following setups for the frequency range:

============================= ==============================================================================================
namespace                     Description
============================= ==============================================================================================
``linear_frequencies``        linear frequency range from ``SI::omega_min`` to ``SI::omega_max`` with ``N_omega`` steps
``log_frequencies``           logarithmic frequency range from ``SI::omega_min`` to ``SI::omega_max`` with ``N_omega`` steps
``frequencies_from_list``     ``N_omega`` frequencies taken from a text file with location ``listLocation[]``
============================= ==============================================================================================



All three options require variable definitions in the according namespaces as described below:

For the **linear frequency** scale all definitions need to be in the ``picongpu::plugins::radiation::linear_frequencies`` namespace. 
The number of total sample frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``.
In the sub-namespace ``SI``, a minimal frequency ``omega_min`` and a maximum frequency ``omega_max`` need to be defined as ``constexpr float_64``.

For the **logarithmic frequency** scale all definitions need to be in the ``picongpu::plugins::radiation::log_frequencies`` namespace. 
Equivalently to the linear case, three variables need to be defined: 
The number of total sample frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``.
In the sub-namespace ``SI``, a minimal frequency ``omega_min`` and a maximum frequency ``omega_max`` need to be defined as ``constexpr float_64``.

For the **file-based frequency** definition,  all definitions need to be in the ``picongpu::plugins::radiation::frequencies_from_list`` namespace.
The number of total frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``  and the path to the file containing the frequency values in units of :math:`\mathrm{[s^{-1}]}` needs to be given as ``constexpr const char * listLocation = "/path/to/frequency_list";``.
The frequency values in the file can be separated by newlines, spaces, tabs, or any other whitespace. The numbers should be given in such a way, that c++ standard ``std::ifstream`` can interpret the number e.g., as ``2.5344e+16``. 

.. note::

   Currently, the variable ``listLocation`` is required to be defined in the ``picongpu::plugins::radiation::frequencies_from_list`` namespace, even if ``frequencies_from_list`` is not used.
   The string does not need to point to an existing file, as long as the file-based frequency definition is not used.


Observation directions
""""""""""""""""""""""

The number of observation directions ``N_theta`` is defined in :ref:`radiation.param <usage-params-plugins>`, but the distribution of observation directions is given in :ref:`radiationObserver.param <usage-params-plugins>`)
There, the function ``observation_direction`` defines the observation directions.

This function returns the x,y and z component of a **unit vector** pointing in the observation direction. 

.. code:: cpp

   DINLINE vector_64
   observation_direction( int const observation_id_extern )
   {
       /* use the scalar index const int observation_id_extern to compute an 
        * observation direction (x,y,y) */
       return vector_64( x , y , z );
   }

.. note::

   The ``radiationObserver.param`` set up will be subject to **further changes**.
   These might be *namespaces* that describe several preconfigured layouts or a functor if *C++ 11* is included in the *nvcc*.


Nyquist limit
"""""""""""""

A major limitation of discrete Fourier transform is the limited frequency resolution due to the discrete time steps of the temporal signal.
(see `Nyquist-Shannon sampling theorem <https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem>`_)
Due to the consideration of relativistic delays, the sampling of the emitted radiation is not equidistantly sampled. 
The plugin has the option to ignore any frequency contributions that lies above the frequency resolution given by the Nyquist-Shannon sampling theorem. 
Because performing this check costs computation time, it can be switched off. 
This is done via a precompiler pragma:

.. code:: cpp

   // Nyquist low pass allows only amplitudes for frequencies below Nyquist frequency
   // 1 = on (slower and more memory, no Fourier reflections)
   // 0 = off (faster but with Fourier reflections)
   #define __NYQUISTCHECK__ 0

Additionally, the maximally resolvable frequency compared to the Nyquist frequency can be set.

.. code:: cpp

   namespace radiationNyquist
   {
       /* only use frequencies below 1/2*Omega_Nyquist */
       const float NyquistFactor = 0.5;
   }

This allows to make a save margin to the hard limit of the Nyquist frequency. 
By using ``NyquistFactor = 0.5`` for periodic boundary conditions, particles that jump from one border to another and back can still be considered. 


Form factor
"""""""""""

The *form factor* is a method, which considers the shape of the macro particles when computing the radiation.
More details can be found in [Pausch2018]_ and [Pausch2019]_.

One can select between different macro particle shapes.
Currently eight shapes are implemented.
A shape can be selected by choosing one of the available namespaces:

.. code:: cpp

   /* choosing the 3D CIC-like macro particle shape */
   namespace radFormFactor = radFormFactor_CIC_3D;


==================================== ===================================================================================================================
Namespace                            Description
==================================== ===================================================================================================================
``radFormFactor_CIC_3D``             3D Cloud-In-Cell shape
``radFormFactor_TSC_3D``             3D Triangular shaped density cloud
``radFormFactor_PCS_3D``             3D Quadratic spline density shape (Piecewise Cubic Spline assignment function)
``radFormFactor_CIC_1Dy``            Cloud-In-Cell shape in y-direction, dot like in the other directions
``radFormFactor_Gauss_spherical``    symmetric Gauss charge distribution
``radFormFactor_Gauss_cell``         Gauss charge distribution according to cell size
``radFormFactor_incoherent``         forces a completely incoherent emission by scaling the macro particle charge with the square root of the weighting
``radFormFactor_coherent``           forces a completely coherent emission by scaling the macro particle charge with the weighting
==================================== ===================================================================================================================


Reducing the particle sample
""""""""""""""""""""""""""""

In order to save computation time, only a random subset of all macro particles can be used to compute the emitted radiation.
In order to do that, the radiating particle species needs the attribute ``radiationMask`` (which is initialized as ``false``) which further needs to be manipulated, to set to true for specific (random) particles.  


.. note::

   The reduction of the total intensity is not considered in the output.
   The intensity will be (in the incoherent case) will be smaller by the fraction of marked to all particles.

.. note::

   The radiation mask is only added to particles, if not all particles should be considered for radiation calculation.
   Adding the radiation flag costs memory.

.. note::

   In future updates, the radiation will only be computed using an extra particle species.
   Therefore, this setup will be subject to further changes.


Gamma filter
""""""""""""

In order to consider the radiation only of particles with a gamma higher than a specific threshold, the radiating particle species needs the attribute ``radiationMask`` (which is initialized as ``false``).
Using a filter functor as:

.. code:: cpp

   using RadiationParticleFilter = picongpu::particles::manipulators::FreeImpl<
       GammaFilterFunctor
    >;

(see Bunch or Kelvin Helmholtz example for details)
sets the flag to true is a particle fulfills the gamma condition.  

.. note::

   More sophisticated filters might come in the near future.
   Therefore, this part of the code might be subject to changes.


Window function filter
""""""""""""""""""""""

A window function can be added to the simulation area to reduce `ringing artifacts <https://en.wikipedia.org/wiki/Ringing_artifacts>`_ due to sharp transition from radiating regions to non-radiating regions at the boundaries of the simulation box.
This should be applied to simulation setups where the entire volume simulated is radiating (e.g. Kelvin-Helmholtz Instability).

In ``radiation.param`` the precompiler variable ``PIC_RADWINDOWFUNCTION`` defines if the window function filter should be used or not.

.. code:: cpp

   // add a window function weighting to the radiation in order
   // to avoid ringing effects from sharp boundaries
   // 1 = on (slower but with noise/ringing reduction)
   // 0 = off (faster but might contain ringing)
   #define PIC_RADWINDOWFUNCTION 0

If set to ``1``, the window function filter is used.

There are several different window function available:

.. code:: cpp

   /* Choose different window function in order to get better ringing reduction
    * radWindowFunctionRectangle
    * radWindowFunctionTriangle
    * radWindowFunctionHamming
    * radWindowFunctionTriplett
    * radWindowFunctionGauss
    */
   namespace radWindowFunctionRectangle { }
   namespace radWindowFunctionTriangle { }
   namespace radWindowFunctionHamming { }
   namespace radWindowFunctionTriplett { }
   namespace radWindowFunctionGauss { }

   namespace radWindowFunction = radWindowFunctionTriangle;
 
By setting ``radWindowFunction`` a specific window function is selected.

More details can be found in [Pausch2019]_.

.cfg file
^^^^^^^^^

For a specific (charged) species ``<species>`` e.g. ``e``, the radiation can be computed by the following commands.  

========================================= ==============================================================================================================================
Command line option                       Description
========================================= ==============================================================================================================================
``--<species>_radiation.period``          Gives the number of time steps between which the radiation should be calculated.
                                          Default is ``0``, which means that the radiation in never calculated and therefor off.
                                          Using ``1`` calculates the radiation constantly. Any value ``>=2`` is currently producing nonsense.
``--<species>_radiation.dump``            Period, after which the calculated radiation data should be dumped to the file system.
                                          Default is ``0``, therefor never.
                                          In order to store the radiation data, a value ``>=1`` should be used.
``--<species>_radiation.lastRadiation``   If set, the radiation spectra summed between the last and the current dump-time-step are stored.
                                          Used for a better evaluation of the temporal evolution of the emitted radiation.
``--<species>_radiation.folderLastRad``   Name of the folder, in which the summed spectra for the simulation time between the last dump and the current dump are stored.
                                          Default is ``lastRad``.
``--<species>_radiation.totalRadiation``  If set the spectra summed from simulation start till current time step are stored.
``--<species>_radiation.folderTotalRad``  Folder name in which the total radiation spectra, integrated from the beginning of the simulation, are stored.
                                          Default ``totalRad``.
``--<species>_radiation.start``           Time step, at which PIConGPU starts calculating the radiation.
                                          Default is ``2`` in order to get enough history of the particles.
``--<species>_radiation.end``             Time step, at which the radiation calculation should end.
                                          Default: ``0`` (stops at end of simulation).
``--<species>_radiation.radPerGPU``       If set, each GPU additionally stores its own spectra without summing over the entire simulation area.
                                          This allows for a localization of specific spectral features.
``--<species>_radiation.folderRadPerGPU`` Name of the folder, where the GPU specific spectra are stored.
                                          Default: ``radPerGPU``
``--<species>_radiation.compression``     If set, the hdf5 output is compressed.
``--<species>_radiation.numJobs``         Number of independent jobs used for the radiation calculation.
                                          This option is used to increase the utilization of the device by producing more independent work.
                                          This option enables accumulation of data in parallel into multiple temporary arrays, thereby increasing the utilization of
                                          the device by increasing the memory footprint
                                          Default: ``2``
========================================= ==============================================================================================================================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

locally, ``numJobs`` times number of frequencies ``N_omega`` times number of directions ``N_theta`` is permanently allocated.
Each result element (amplitude) is a double precision complex number.

Host
""""

as on accelerator.

Output
^^^^^^

Depending on the command line options used, there are different output files.

======================================== ========================================================================================================================
Command line flag                        Output description
======================================== ========================================================================================================================
``--<species>_radiation.totalRadiation`` Contains *ASCII* files that have the total spectral intensity until the timestep specified by the filename.
                                         Each row gives data for one observation direction (same order as specified in the ``observer.py``).
                                         The values for each frequency are separated by *tabs* and have the same order as specified in ``radiation.param``.
                                         The spectral intensity is stored in the units :math:`\mathrm{[Js]}`.
``--<species>_radiation.lastRadiation``  has the same format as the output of *totalRadiation*.
                                         The spectral intensity is only summed over the last radiation ``dump`` period.
``--<species>_radiation.radPerGPU``      Same output as *totalRadiation* but only summed over each GPU. 
                                         Because each GPU specifies a spatial region, the origin of radiation signatures can be distinguished.
*radiationHDF5*                          In the folder  ``radiationHDF5``, hdf5 files for each radiation dump and species are stored.
                                         These are complex amplitudes in units used by *PIConGPU*.
                                         These are for restart purposes and for more complex data analysis.
======================================== ========================================================================================================================


Text-based output
"""""""""""""""""

The text-based output of ``lastRadiation`` and ``totalRadiation`` contains the intensity values in SI-units :math:`\mathrm{[Js]}`. Intensity values for different frequencies are separated by spaces, while newlines separate values for different observation directions. 


In order to read and plot the text-based radiation data, a python script as follows could be used:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # frequency definition:
    # as defined in the 'radiation.param' file:
    N_omega = 1024
    omega_min = 0.0 # [1/s]
    omega_max = 5.8869e17 # [1/s]
    omega = np.linspace(omega_min, omega_max, N_omega)

    # observation angle definition:
    # as defined in the 'radiation.param' file:
    N_observer = 128
    # as defined in the 'radiationObserver.param' file:
    # this example assumes one used the default Bunch example
    # there, the theta values are normalized to the Lorentz factor
    theta_min = -1.5 # [rad/gamma]
    theta_max = +1.5 # [rad/gamma]
    theta = np.linspace(theta_min, theta_max, N_observer)

    # load radiation text-based data
    rad_data = np.loadtxt('./simOutput/lastRad/e_radiation_2820.dat')

    # plot radiation spectrum
    plt.figure()
    plt.pcolormesh(omega, theta, rad_data, norm=LogNorm())

    # add and configure colorbar
    cb = plt.colorbar()
    cb.set_label(r"$\frac{\mathrm{d}^2 I}{\mathrm{d} \omega \mathrm{d} \Omega} \, \mathrm{[Js]}$", fontsize=18)
    for i in cb.ax.get_yticklabels():
        i.set_fontsize(14)

    # configure x-axis
    plt.xlabel(r"$\omega \, \mathrm{[1/s]}$", fontsize=18)
    plt.xticks(fontsize=14)

    # configure y-axis
    plt.ylabel(r"$\theta / \gamma$", fontsize=18)
    plt.yticks(fontsize=14)

    # make plot look nice
    plt.tight_layout()
    plt.show()


HDF5 output
"""""""""""

The hdf5 based data contains the following data structure in ``/data/{iteration}/DetectorMesh/`` according to the openPMD standard:

**Amplitude (Group):**

======== ===================================================== ====================================
Dataset  Description                                           Dimensions
======== ===================================================== ====================================
``x_Re`` real part, x-component of the complex amplitude       (``N_observer``, ``N_omega``, 1)
``x_Im`` imaginary part, x-component of the complex amplitude  (``N_observer``, ``N_omega``, 1)
``y_Re`` real part, y-component of the complex amplitude       (``N_observer``, ``N_omega``, 1)
``y_Im`` imaginary part, y-component of the complex amplitude  (``N_observer``, ``N_omega``, 1)
``z_Re`` real part, z-component of the complex amplitude       (``N_observer``, ``N_omega``, 1)
``z_Im`` imaginary part, z-component of the complex amplitude  (``N_observer``, ``N_omega``, 1)
======== ===================================================== ====================================

.. note::

   Please be aware, that despite the fact, that the SI-unit of each amplitude entry is :math:`\mathrm{[\sqrt{Js}]}`, the stored ``unitSI`` attribute returns :math:`\mathrm{[Js]}`.
   This inconsistency will be fixed in the future.
   Until this inconstincy is resolved, please multiply the datasets with the square root of the ``unitSI`` attribute to convert the amplitudes to SI units. 
   

**DetectorDirection (Group):**

======== ======================================================= ===============================
Dataset  Description                                             Dimensions
======== ======================================================= ===============================
``x``    x-component of the observation direction :math:`\vec n` (``N_observer``, 1, 1)
``y``    y-component of the observation direction :math:`\vec n` (``N_observer``, 1, 1)
``z``    z-component of the observation direction :math:`\vec n` (``N_observer``, 1, 1)
======== ======================================================= ===============================

**DetectorFrequency (Group):**

========== ======================================================= ===============================
Dataset    Description                                             Dimensions
========== ======================================================= ===============================
``omega``  frequency :math:`\omega` of virtual detector bin        (1, ``N_omega``, 1)
========== ======================================================= ===============================


Please be aware that all datasets in the hdf5 output are given in the PIConGPU-intrinsic unit system. In order to convert, for example, the frequencies :math:`\omega` to SI-units one has to multiply with the dataset-attribute `unitSI`. 

.. code:: python

   import h5py
   f = h5py.File("e_radAmplitudes_2800_0_0_0.h5", "r")
   omega_handler = f['/data/2800/DetectorMesh/DetectorFrequency/omega']
   omega = omega_handler[0, :, 0] * omega_handler.attrs['unitSI'] 
   f.close()

In order to extract the radiation data from the HDF5 datasets, PIConGPU provides a python module to read the data and obtain the result in SI-units. An example python script is given below:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt 
    from matplotlib.colors import LogNorm

    from picongpu.plugins.data import RadiationData

    # access HDF5 radiation file
    radData = RadiationData("./simOutput/radiationHDF5/e_radAmplitudes_2820_0_0_0.h5")

    # get frequencies
    omega = radData.get_omega()

    # get all observation vectors and convert to angle

    vec_n = radData.get_vector_n()
    gamma = 5.0
    theta_norm = np.arctan(vec_n[:, 0]/vec_n[:, 1]) * gamma 

    # get spectrum over observation angle
    spectrum = radData.get_Spectra()

    # plot radiation spectrum
    plt.figure()
    plt.pcolormesh(omega, theta_norm, spectrum, norm=LogNorm())

    # add and configure colorbar
    cb = plt.colorbar()
    cb.set_label(r"$\frac{\mathrm{d}^2 I}{\mathrm{d} \omega \mathrm{d} \Omega} \, \mathrm{[Js]}$", fontsize=18)
    for i in cb.ax.get_yticklabels():
        i.set_fontsize(14)

    # configure x-axis
    plt.xlabel(r"$\omega \, \mathrm{[1/s]}$", fontsize=18)
    plt.xticks(fontsize=14)

    # configure y-axis
    plt.ylabel(r"$\theta / \gamma$", fontsize=18)
    plt.yticks(fontsize=14)

    # make plot look nice
    plt.tight_layout()
    plt.show()


There are various methods besides ``get_Spectra()`` that are provided by the python module.
If a method exists for ``_x`` (or ``_X``) it also exists for ``_y`` and ``_z`` (``_Y`` and ``_Z``) accordingly.

============================ ==============================================================================================================
Method                       Description
============================ ==============================================================================================================
``.get_omega()``             get frequency :math:`\omega` of virtual detector bin in units of :math:`\mathrm{[1/s]}`
``.get_vector_n()``          get observation direction :math:`\vec{n}`
``.get_Spectra()``           get spectrum :math:`\mathrm{d}^2 I / \mathrm{d} \omega \mathrm{d} \Omega` in units of :math:`\mathrm{[Js]}`
``.get_Polarization_X()``    get spectrum but only for polarization in x-direction
``.get_Amplitude_x()``       get x-component of complex amplitude (unit: :math:`\mathrm{[\sqrt{Js}]}`)
``.get_timestep()``          the iteration (timestep) at which the data was produced (unit: PIC-cycles)
============================ ==============================================================================================================

.. note::

   Modules for visualizing radiation data and a widget interface to explore the data interactively will be developed in the future. 

Analyzing tools
^^^^^^^^^^^^^^^

In ``picongp/src/tools/bin``, there are tools to analyze the radiation data after the simulation.

============================== ======================================================================================================================================
Tool                           Description
============================== ======================================================================================================================================
``plotRadiation``              Reads *ASCII* radiation data and plots spectra over angles as color plots.
                               This is a python script that has its own help.
                               Run ``plotRadiation --help`` for more information.
``radiationSyntheticDetector`` Reads *ASCII* radiation data and statistically analysis the spectra for a user specified region of observation angles and frequencies.
                               This is a python script that has its own help. Run ``radiationSyntheticDetector --help`` for more information.
*smooth.py*                    Python module needed by ``plotRadiation``.
============================== ======================================================================================================================================


Known Issues
^^^^^^^^^^^^

The plugin supports multiple radiation species but spectra (frequencies and observation directions) are the same for all species. 


References
^^^^^^^^^^

.. [Pausch2012]
       Pausch, R.
       *Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics*
       Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2012)
       https://doi.org/10.5281/zenodo.843510

.. [Pausch2014]
       Pausch, R., Debus, A., Widera, R. et al.
       *How to test and verify radiation diagnostics simulations within particle-in-cell frameworks*
       Nuclear Instruments and Methods in Physics Research, Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 740, 250–256 (2014)
       https://doi.org/10.1016/j.nima.2013.10.073

.. [Pausch2018]
       Pausch, R., Debus, A., Huebl, A. at al.
       *Quantitatively consistent computation of coherent and incoherent radiation in particle-in-cell codes — A general form factor formalism for macro-particles*
       Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 909, 419–422 (2018)
       https://doi.org/10.1016/j.nima.2018.02.020

.. [Pausch2019]
       Pausch, R.
       *Synthetic radiation diagnostics as a pathway for studying plasma dynamics from advanced accelerators to astrophysical observations*
       PhD Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019)
       https://doi.org/10.5281/zenodo.3616045



