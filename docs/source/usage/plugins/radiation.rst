.. _usage-plugins-radiation:

Radiation
---------

The spectrally resolved far field radiation of charged macro particles.

Our simulation computes the `Lienard Wiechert potentials <https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential>`_ to calculate the emitted electromagnetic spectra for different observation directions using the far field approximation.

.. math::

   \frac{\operatorname{d}^2I}{\operatorname{d}{\Omega}\operatorname{d}\omega}\left(\omega,\vec{n}\right)=\left|\sum\limits_{k=1}^{N}\int\limits_{-\infty}^{+\infty}\frac{\vec{n}\times\left[\left(\vec{n}-\vec{\beta}_k(t)\right)\times\dot{\vec{\beta}}_k(t)\right]}{\left(1-\vec{\beta}_k(t)\cdot\vec{n}\right)^2}\cdot\operatorname{e}^{\operatorname{i}\omega\left(t-\vec{n}\cdot\vec{r}_k(t)/c\right)}\operatorname{d}t\right|^2

============================== ================================================================================
Variable                       Meaning
============================== ================================================================================
:math:`\vec r_k(t)`            The position of particle *k* at time *t*.
:math:`\vec \beta_k(t)`        The normalized speed of particle *k* at time *t*.
                               (Speed devided by the speed of light)
:math:`\dot{\vec{\beta}}_k(t)` The normalized acceleration of particle *k* at time *t*.
                               (Time derivative of the normalized speed.)
:math:`t`                      Time
:math:`\vec n`                 Unit vector pointing in the direction where the far field radiation is observed.
:math:`\omega`                  The circular frequency of the radiation that is observed.
:math:`N`                      Number of all (macro) particles that are used for computing the radiation.
:math:`k`                      Running index of the particles.
============================== ================================================================================

Currently this allows to predict the emitted radiation from plasmas if it can be described by classical means.
Not considered are emissions from ionization, Compton scattering or any bremsstrahlung that originate from scattering on scales smaller than the PIC cell size. 

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.param files
^^^^^^^^^^^^

In order to setup the radiation analyzer plugin, both the :ref:`radiation.param <usage-params-plugins>` and the :ref:`radiationObserver.param <usage-params-plugins>` have to be configured **and** the radiating particles need to have the attribute ``momentumPrev1`` which can be added in :ref:`speciesDefinition.param <usage-params-core>`.

In *radiationConfig.param*, the number of frequencies ``N_omega`` and observation directions ``N_theta`` is defined.

Frequency range
"""""""""""""""

The frequency range is set up by choosing a specific namespace that defines the frequency setup

.. code:: cpp

   /* choose linear frequency range */
   namespace radiation_frequencies = rad_linear_frequencies;

Currently you can choose from the following setups for the frequency range:

============================= ==============================================================================================
namespace                     Description
============================= ==============================================================================================
``rad_linear_frequencies``    linear frequency range from ``SI::omega_min`` to ``SI::omega_max`` with ``N_omega`` steps
``rad_log_frequencies``       logarithmic frequency range from ``SI::omega_min`` to ``SI::omega_max`` with ``N_omega`` steps
``rad_frequencies_from_list`` ``N_omega`` frequencies taken from a text file with location ``listLocation[]``
============================= ==============================================================================================


Observation directions
""""""""""""""""""""""

The number of observation directions `N_theta` is defined in :ref:`radiation.param <usage-params-plugins>`, but the distribution of observation directions is given in :ref:`radiationObserver.param.param <usage-params-plugins>`)
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

The *form factor* is still an experimental method trying to consider the shape of the macro particles when computing the radiation.
By default, it should be switched off by setting ``__COHERENTINCOHERENTWEIGHTING__`` to zero. 

.. code:: cpp

   // corect treatment of coherent and incoherent radiation from macroparticles
   // 1 = on (slower and more memory, but correct quantitative treatment)
   // 0 = off (faster but macroparticles are treated as highly charged, point-like particle)
   #define __COHERENTINCOHERENTWEIGHTING__ 0


If switched on, one can select between different macro particle shapes. 
Currently three shapes are implemented.
A shape can be selected by choosing one of the available namespaces:

.. code:: cpp

   /* choosing the 3D CIC-like macro particle shape */
   namespace radFormFactor_selected = radFormFactor_CIC_3D;


============================ ===================================================================================================================
Namespace                    Description
============================ ===================================================================================================================
``radFormFactor_CIC_3D``     3D Cloud-In-Cell shape
``radFormFactor_CIC_1Dy``    Cloud-In-Cell shape in y-direction, dot like in the other directions
``radFormFactor_incoherent`` forces a completely incoherent emission by scaling the macro particle charge with the square root of the weighting
============================ ===================================================================================================================

.. note:

   possibly more shapes (f.e. spaghetti shape) will be added


Reducing the particle sample
""""""""""""""""""""""""""""

In order to save computation time, only a random subset of all macro particles can be used to compute the emitted radiation.
In order to do that, the radiating particle species needs the attribute ``radiationMask`` (which is initialized as ``false``) which further needs to be manipulated, to set to true for specific (random) particles.  


.. note::

   The reduction of the total intensity is not considered in the output.
   The intensity will be (in the incoherent case) by the fraction of marked marticles smaller than in the case of selecting all particles.

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

In ``radiationConfig.param`` the precompiler variable ``PIC_RADWINDOWFUNCTION`` defines if the window function filter should be used or not.

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


.cfg file
^^^^^^^^^

For a specific (charged) species ``<species>`` e.g. ``e``, the radiation can be computed by the following commands.  

========================================= ==============================================================================================================================
Command line option                       Description
========================================= ==============================================================================================================================
``--radiation_<species>.period``          Gives the number of time steps between which the radiation should be calculated.
                                          Default is ``0``, which means that the radiation in never calculated and therefor off.
                                          Using `1` calculates the radiation constantly. Any value ``>=2`` is currently producing nonsense.
``--radiation_<species>.dump``            Period, after which the calculated radiation data should be dumped to the file system.
                                          Default is ``0``, therefor never.
                                          In order to store the radiation data, a value `>=1` should be used.
``--radiation_<species>.lastRadiation``   If set, the radiation spectra summed between the last and the current dump-time-step are stored.
                                          Used for a better evaluation of the temporal evolution of the emitted radiation.
``--radiation_<species>.folderLastRad``   Name of the folder, in which the summed spectra for the simulation time between the last dump and the current dump are stored.
                                          Default is ``lastRad``.
``--radiation_<species>.totalRadiation``  If set the spectra summed from simulation start till current time step are stored.
``--radiation_<species>.folderTotalRad``  Folder name in which the total radiation spectra, integrated from the beginning of the simulation, are stored.
                                          Default ``totalRad``.
``--radiation_<species>.start``           Time step, at which PIConGPU starts calculating the radiation.
                                          Default is ``2`` in order to get enough history of the particles.
``--radiation_<species>.end``             Time step, at which the radiation calculation should end.
                                          Default: `0`(stops at end of simulation).
``--radiation_<species>.omegaList``       In case the frequencies for the spectrum are coming from a list stored in a file, this gives the path to this list.
                                          Default: `_noPath_` throws an error. *This does not switch on the frequency calculation via list.*
``--radiation_<species>.radPerGPU``       If set, each GPU additionally stores its own spectra without summing over the entire simulation area.
                                          This allows for a localization of specific spectral features.
``--radiation_<species>.folderRadPerGPU`` Name of the folder, where the GPU specific spectra are stored.
                                          Default: ``radPerGPU``
``--radiation_<species>.compression``     If set, the hdf5 output is compressed.
========================================= ==============================================================================================================================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

each energy bin times each coordinate bin allocates one counter (``float_X``) permanently and on each accelerator.

Host
""""

as on accelerator.

Output
^^^^^^

Depending on the command line options used, there are different output files.

======================================== ========================================================================================================================
Command line flag                        Output description
======================================== ========================================================================================================================
``--radiation_<species>.totalRadiation`` Contains *ASCII* files that have the total spectral intensity until the timestep specified by the filename.
                                         Each row gives data for one observation direction (same order as specified in the ``observer.py``).
                                         The values for each frequency are separated by *tabs* and have the same order as specified in ``radiationConfig.param``.
                                         The spectral intensity is stored in the units **[J s]**.
``--radiation_<species>.lastRadiation``  has the same format as the output of *totalRadiation*.
                                         The spectral intensity is only summed over the last radiation `dump` period.
``--radiation_<species>.radPerGPU``      Same output as *totalRadiation* but only summed over each GPU. 
                                         ecause each GPU specifies a spatial region, the origin of radiation signatures can be distinguished.
*radiationHDF5*                          In the folder  ``radiationHDF5``, hdf5 files for each radiation dump and species are stored.
                                         These are complex amplitudes in units used by *PIConGPU*.
                                         These are for restart purposes and for more complex data analysis.
======================================== ========================================================================================================================

Analysing tools
^^^^^^^^^^^^^^^^

In ``picongp/src/tools/bin``, there are tools to analyze the radiation data after the simulation.

============================== ======================================================================================================================================
Tool                           Description
============================== ======================================================================================================================================
``plotRadiation``              Reads *ASCII* radiation data and plots spectra over angles as color plots.
                               This is a python script that has its own help.
                               Run ``plotRadiation --help`` for more information.
``radiationSyntheticDetector`` Reads *ASCII* radiation data and statistically analysis the spectra for a user specified region of observation angles and frequencies.
                               This is a python script that has its own help. Run ``radiationSyntheticDetector --help`` for more informations.
*smooth.py*                    Python module needed by `plotRadiation`.
============================== ======================================================================================================================================


Known Issues
^^^^^^^^^^^^

Currently, the radiation plugin does not support 2D simulations. 
This should be fixed with `issue #289 <https://github.com/ComputationalRadiationPhysics/picongpu/issues/289>`_ .
The plugin supports multiple radiation species but spectra (frequencies and observation directions) are the same for all species. 


References
^^^^^^^^^^

- `Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics <https://www.hzdr.de/db/Cms?pOid=38997>`_,
  Diploma thesis on the radiation plugin
- `How to test and verify radiation diagnostics simulations within particle-in-cell frameworks <http://dx.doi.org/10.1016/j.nima.2013.10.073>`_,
  Some tests that have been performed to validate the code
