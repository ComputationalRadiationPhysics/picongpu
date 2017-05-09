.. _usage-workflows-quasiNeutrality:

Quasi-Neutral Initialization
----------------------------

.. sectionauthor:: Axel Huebl

In order to initialize the electro-magnetic fields self-consistently, one needs to fulfill Gauss's law :math:`\vec \nabla \cdot \vec E = \frac{\rho}{\epsilon_0}` (and :math:`\vec \nabla \cdot \vec B = 0`).
The trivial solution to this equation is to start *field neutral* by microscopically placing a charge-compensating amount of free electrons on the same position as according ions.

Fully Ionized Ions
""""""""""""""""""

For fully ionized ions, just use ``ManipulateDeriveSpecies`` in :ref:`speciesInitialization.param <usage-params-core>` and derive macro-electrons :math:`1:1` from macro-ions but increase their weighting by :math:`1:Z` of the ion.

.. code-block:: cpp

   using InitPipeline = mpl::vector<
       // density profile from density.param and
       //     start position from particle.param
       CreateDensity<
           densityProfiles::YourSelectedProfile,
           startPosition::YourStartPosition,
           Carbon
       >,
       // create a macro electron for each macro carbon but increase its
       //     weighting by the ion's proton number so it represents all its
       //     electrons after an instantanous ionization
       ManipulateDeriveSpecies<
           manipulators::ProtonTimesWeighting,
           Carbon,
           Electrons
       >
   >;

Partly Ionized Ions
"""""""""""""""""""

For partial pre-ionization, the :ref:`FoilLCT example <usage-examples-foilLCT>` shows a detailed setup.
First, define a functor that manipulates the number of bound electrons in :ref:`particle.param <usage-params-core>`, e.g. to *once ionized*.
Then again in :ref:`speciesInitialization.param <usage-params-core>` set your initialization routines to:

.. code-block:: cpp

   using InitPipeline = mpl::vector<
       // density profile from density.param and
       //     start position from particle.param
       CreateDensity<
           densityProfiles::YourSelectedProfile,
           startPosition::YourStartPosition,
           Carbon
       >,
       // partially pre-ionize the carbons by manipulating the carbon's
       //     `boundElectrons` attribute,
       //     functor defined in particle.param: set to C1+
       Manipulate<
           manipulators::OnceIonized,
           Carbon
       >,
       // does not manipulate the weighting while deriving the electrons
       //     ("once pre-ionized") since we set carbon as C1+
       DeriveSpecies<
           Carbon,
           Electrons
       >
   >;

If you want to initialize an arbitrary ionization state, just add your own weighting manipulating functor in :ref:`particle.param <usage-params-core>` for the electrons and use it with ``ManipulateDeriveSpecies`` as in the first example.

In the first example, which does not manipulate the ``boundElectrons`` attribute of the carbon species (optional, see :ref:`speciesAttributes.param <usage-params-core>`), the default is used (zero bound electrons, fully ionized).
