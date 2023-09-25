.. _usage-plugins-transitionRadiation:

Transition Radiation
--------------------

The spectrally resolved far field radiation created by electrons passing through a metal foil.

Our simulation computes the `transition radiation <https://en.wikipedia.org/wiki/Transition_radiation>`_ to calculate the emitted electromagnetic spectra for different observation angles.


.. math::

   \frac{d^2W}{d\omega d\Omega} = \frac{e^2 N_e}{(4 \pi \epsilon_0)\pi^2 c}\Bigg\{ \bigg[ \int d^3 \vec{p} g(\mathcal{E}^2_\parallel + \mathcal{E}^2_\perp) \bigg] + \big(N_e - 1\big) \bigg[ \Big| \int d^3 \vec{p} g \mathcal{E}_\parallel F \Big|^2 + \Big| \int d^3 \vec{p} g \mathcal{E}_\perp F \Big|^2\bigg] \Bigg\}

.. math::

   \mathcal{E}_\parallel = \frac{u \cos \psi \Big[ u \sin\psi \cos\phi - (1+u^2)^{1/2} \sin\theta \Big]}{\mathcal{N}(\theta, u, \psi, \phi)}

.. math::

   \mathcal{E}_\perp = \frac{u^2 \cos \psi \sin\psi \sin\phi  \cos\theta}{\mathcal{N}(\theta, u, \psi, \phi)}

.. math::

   \mathcal{N}(\theta, u, \psi, \phi) = \Big[ \big(1+u^2\big)^{1/2} - u \sin\psi \cos\phi \sin\theta\Big]^2 - u^2 \cos^2\psi \cos^2\theta

.. math::

   F = \frac{1}{g(\vec{p})} \int d^2 \vec{r}_\perp e^{-i\vec{k}_\perp \cdot \vec{r}_\perp} \int dy e^{-i y (\omega - \vec{k}_\perp \cdot \vec{v}_\perp) / v_y} h(\vec{r}, \vec{p})

============================== ================================================================================
Variable                       Meaning
============================== ================================================================================
:math:`N_e`                    Amount of real electrons
:math:`\psi`                   Azimuth angle of momentum vector from electrons to y-axis of simulation
:math:`\theta`                 Azimuth angle of observation vector
:math:`\phi`                   Polar angle between momentum vector from electrons and observation vector
:math:`\omega`                 The circular frequency of the radiation that is observed.
:math:`h(\vec{r}, \vec{p})`     Normalized phasespace distribution of electrons
:math:`g(\vec{p})`             Normalized momentum distribution of electrons
:math:`g(\vec{p})`             Normalized momentum distribution of electrons
:math:`\vec{k}`                Wavevector of electrons
:math:`\vec{v}`                Velocity vector of electrons
:math:`u`                      Normalized momentum of electrons :math:`\beta \gamma`
:math:`\mathcal{E}`            Normalized energy of electrons
:math:`\mathcal{N}`            Denominator of normalized energies
:math:`F`                      Normalized formfactor of electrons, contains phase informations
============================== ================================================================================

This plugin allows to predict the emitted virtual transition radiation, which would be caused by the electrons in the simulation box passing through a virtual metal foil which is set at a specific location.
The transition radiation can only be calculated for electrons at the moment.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`openPMD API <install-dependencies>` is compiled in.

.param files
^^^^^^^^^^^^

In order to setup the transition radiation plugin, the :ref:`transitionRadiation.param <usage-params-plugins>` has to be configured **and** the radiating particles need to have the attributes ``weighting``, ``momentum``, ``location``, and ``transitionRadiationMask`` (which can be added in :ref:`speciesDefinition.param <usage-params-core>`) as well as the flags ``massRatio`` and ``chargeRatio``.

In *transitionRadiation.param*, the number of frequencies ``N_omega`` and observation directions ``N_theta`` and ``N_phi`` are defined.

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

For the **linear frequency** scale all definitions need to be in the ``picongpu::plugins::transitionRadiation::linear_frequencies`` namespace.
The number of total sample frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``.
In the sub-namespace ``SI``, a minimal frequency ``omega_min`` and a maximum frequency ``omega_max`` need to be defined as ``constexpr float_64``.

For the **logarithmic frequency** scale all definitions need to be in the ``picongpu::plugins::transitionRadiation::log_frequencies`` namespace.
Equivalently to the linear case, three variables need to be defined: 
The number of total sample frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``.
In the sub-namespace ``SI``, a minimal frequency ``omega_min`` and a maximum frequency ``omega_max`` need to be defined as ``constexpr float_64``.

For the **file-based frequency** definition,  all definitions need to be in the ``picongpu::plugins::transitionRadiation::frequencies_from_list`` namespace.
The number of total frequencies ``N_omega`` need to be defined as ``constexpr unsigned int``  and the path to the file containing the frequency values in units of :math:`[s^{-1}]` needs to be given as ``constexpr const char * listLocation = "/path/to/frequency_list";``.
The frequency values in the file can be separated by newlines, spaces, tabs, or any other whitespace. The numbers should be given in such a way, that c++ standard ``std::ifstream`` can interpret the number e.g., as ``2.5344e+16``. 

.. note::

   Currently, the variable ``listLocation`` is required to be defined in the ``picongpu::plugins::transitionRadiation::frequencies_from_list`` namespace, even if ``frequencies_from_list`` is not used.
   The string does not need to point to an existing file, as long as the file-based frequency definition is not used.


Observation directions
""""""""""""""""""""""

The number of observation directions ``N_theta`` and the distribution of observation directions is defined in :ref:`transitionRadiation.param <usage-params-plugins>`.
There, the function ``observationDirection`` defines the observation directions.

This function returns the x,y and z component of a **unit vector** pointing in the observation direction. 

.. code:: cpp

   DINLINE vector_64
   observationDirection( int const observation_id_extern )
   {
       /* use the scalar index const int observation_id_extern to compute an 
        * observation direction (x,y,y) */
       return vector_64( x , y , z );
   }

.. note::

   The ``transitionRadiation.param`` set up will be subject to **further changes**, since the ``radiationObserver.param`` it is based on is subject to further changes.
   These might be *namespaces* that describe several preconfigured layouts or a functor if *C++ 11* is included in the *nvcc*.


Foil Position
"""""""""""""

If one wants to virtually propagate the electron bunch to a foil in a further distance to get a rough estimate of the effect of the divergence on the electron bunch, one can include a foil position.
A foil position which is unequal to zero, adds the electrons momentum vectors onto the electron until they reach the given y-coordinate.
To contain the longitudinal information of the bunch, the simulation window is actually virtually moved to the foil position and not each single electron.
This is a run time parameter which can be set with ``--<species>_transRad.foilPositionY``.

.. note::

    This is an experimental feature, which was not verified yet.

Macro-particle form factor
""""""""""""""""""""""""""

The *macro-particle form factor* is a method, which considers the shape of the macro particles when computing the radiation.

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

.. note::
  One should not confuse this macro-particle form factor with the form factor :math:`F`, which was previously mentioned.
  This form factor is equal to the macro-particle shape, while :math:`F` contains the phase information of the whole electron bunch.
  Both are necessary for a physically correct transition radiation calculation.


Gamma filter
""""""""""""

In order to consider the radiation only of particles with a gamma higher than a specific threshold.
In order to do that, the radiating particle species needs the flag ``transitionRadiationMask`` (which is initialized as ``false``) which further needs to be manipulated, to set to true for specific (random) particles.

Using a filter functor as:

.. code:: cpp

    using GammaFilter = picongpu::particles::manipulators::generic::Free<
        GammaFilterFunctor
    >;

(see TransitionRadiation example for details)
sets the flag to true if a particle fulfills the gamma condition.

.. note::

   More sophisticated filters might come in the near future.
   Therefore, this part of the code might be subject to changes.


.cfg file
^^^^^^^^^

For a specific (charged) species ``<species>`` e.g. ``e``, the radiation can be computed by the following commands.  

========================================== ==============================================================================================================================
Command line option                        Description
========================================== ==============================================================================================================================
``--<species>_transRad.period``            Gives the number of time steps between which the radiation should be calculated.
``--<species>_transRad.foilPositionY``     Absolute position in SI units to put a virtual foil for calculating the transition radiation. See above for more information. Disabled = 0.0. Default: 0.0
``--<species>_transRad.file``              File name to stodre transition radiation in. Default: transRad
``--<species>_transRad.ext``               openPMD filename extension. Default: ext
``--<species>_transRad.datOutput``         Optional text file output in numpy-readable format. Enabled = 1. Default: 0
========================================== ==============================================================================================================================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

two counters (``float_X``) and two counters (``complex_X``) are allocated permanently

Host
""""

as on accelerator.

Output
^^^^^^
Contains *ASCII* files in ``simOutput/transRad`` that have the total spectral intensity until the timestep specified by the filename.
Each row gives data for one observation direction (same order as specified in the ``observer.py``).
The values for each frequency are separated by *tabs* and have the same order as specified in ``transitionRadiation.param``.
The spectral intensity is stored in the units **[J s]**.

Analysing tools
^^^^^^^^^^^^^^^^
The ``transition_radiation_visualizer.py`` in ``lib/python/picongpu/extra/plugins/plot_mpl`` can be used to analyze the radiation data after the simulation.
See ``transition-radiation_visualizer.py --help`` for more information.
It only works, if the input frequency are on a divided logarithmically!

Known Issues
^^^^^^^^^^^^

The output is currently only physically correct for electron passing through a metal foil.

References
^^^^^^^^^^

- *Theory of coherent transition radiation generated at a plasma-vacuum interface*
   Schroeder, C. B. and Esarey, E. and van Tilborg, J. and Leemans, W. P.,
   American Physical Society(2004),
   https://link.aps.org/doi/10.1103/PhysRevE.69.016501

- *Diagnostics for plasma-based electron accelerators*
   Downer, M. C. and Zgadzaj, R. and Debus, A. and Schramm, U. and Kaluza, M. C.,
   American Physical Society(2018),
   https://link.aps.org/doi/10.1103/RevModPhys.90.035002

- *Synthetic characterization of ultrashort electron bunches using transition radiation*
   Carstens, F.-O.,
   Bachelor thesis on the transition radiation plugin,
   https://doi.org/10.5281/zenodo.3469663

- *Quantitatively consistent computation of coherent and incoherent radiation in particle-in-cell codes — A general form factor formalism for macro-particles*
   Pausch, R.,
   Description for the effect of macro-particle shapes in particle-in-cell codes,
   https://doi.org/10.1016/j.nima.2018.02.020
