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
       /* density profile from density.param and
        *     start position from particle.param */
       CreateDensity<
           densityProfiles::YourSelectedProfile,
           startPosition::YourStartPosition,
           Carbon
       >,
       /* create a macro electron for each macro carbon but increase its
        *     weighting by the ion's proton number so it represents all its
        *     electrons after an instantanous ionization */
       ManipulateDeriveSpecies<
           manipulators::ProtonTimesWeighting,
           Carbon,
           Electrons
       >
   >;

If the ``Carbon`` species in this example has an attribute ``boundElectrons`` (optional, see :ref:`speciesAttributes.param and speciesDefinition.param <usage-params-core>`) and its value is not manipulated the default value is used (zero bound electrons, fully ionized).
If the attribute ``boundElectrons`` is not added to the ``Carbon`` species the charge state is considered constant and taken from the ``chargeRatio< ... >`` particle flag.

Partly Ionized Ions
"""""""""""""""""""

For partial pre-ionization, the :ref:`FoilLCT example <usage-examples-foilLCT>` shows a detailed setup.
First, define a functor that manipulates the number of bound electrons in :ref:`particle.param <usage-params-core>`, e.g. to *twice pre-ionized*.

.. code-block:: cpp

   #include "picongpu/particles/traits/GetAtomicNumbers.hpp"
   // ...

   namespace manipulators
   {
       //! ionize ions twice
       struct TwiceIonizedImpl
       {
           template< typename T_Particle >
           DINLINE void operator()(
               T_Particle& particle
           )
           {
               constexpr float_X protonNumber =
                   GetAtomicNumbers< T_Particle >::type::numberOfProtons;
               particle[ boundElectrons_ ] = protonNumber - float_X( 2. );
           }
       };

       //! definition of TwiceIonizedImpl manipulator
       using TwiceIonized = generic::Free< TwiceIonizedImpl >;

   } // namespace manipulators

Then again in :ref:`speciesInitialization.param <usage-params-core>` set your initialization routines to:

.. code-block:: cpp

   using InitPipeline = mpl::vector<
       /* density profile from density.param and
        *     start position from particle.param */
       CreateDensity<
           densityProfiles::YourSelectedProfile,
           startPosition::YourStartPosition,
           Carbon
       >,
       /* partially pre-ionize the carbons by manipulating the carbon's
        *     `boundElectrons` attribute,
        *     functor defined in particle.param: set to C2+ */
       Manipulate<
           manipulators::TwiceIonized,
           Carbon
       >,
       /* does also manipulate the weighting x2 while deriving the electrons
        *     ("twice pre-ionized") since we set carbon as C2+ */
       ManipulateDeriveSpecies<
           manipulators::binary::UnboundElectronsTimesWeighting,
           Carbon,
           Electrons
       >
   >;
