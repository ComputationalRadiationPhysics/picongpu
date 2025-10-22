.. _fusionReactions:

Fusion Workflow
===============

This page explains how to configure and run the Fusion Extension in simulations.
For the physics background, algorithms, and references, see the model page:
:doc:`../../models/fusion`.

Species Definition Requirements
-------------------------------
All species participating in fusion reactions must be defined with ``massRatio`` and ``chargeRatio`` flags. These are mandatory for the fusion algorithms to function correctly.

To simplify defining particle masses relative to the electron mass, PIConGPU provides a constant for the atomic mass unit (``amu``). This constant is the ratio of one atomic mass unit to the electron rest mass. You can use it in ``speciesDefinition.param`` as follows:

.. code-block:: cpp

    #include "picongpu/unitless/simulation.unitless"

    // Atomic mass unit in electron masses (amu/me)
    constexpr float_X amu = 1.0_X / sim.fusion.electronMassAMU;

    // Deuteron species
    value_identifier(float_X, MassRatioDeuterons, 2.013553212745*amu);
    value_identifier(float_X, ChargeRatioDeuterons, 1.0);

    using ParticleFlagsDeuterons = MakeSeq_t<
        particlePusher<UsedParticlePusher>,
        shape<UsedParticleShape>,
        interpolation<UsedField2Particle>,
        massRatio<MassRatioDeuterons>,        // Required for fusion
        chargeRatio<ChargeRatioDeuterons>     // Required for fusion
    >;

    using PIC_Deuterons = Particles<PMACC_CSTRING("d"), ParticleFlagsDeuterons, DefaultParticleAttributes>;

Important: Mass ratios must be as precise as possible since fusion energy release is calculated from the mass deficit between reactants and products.

Fusion Reaction Configuration (Summary)
--------------------------------------
Configure fusion reactions in ``include/picongpu/param/fusion.param``.

1) Define reaction parameters

.. code-block:: cpp

    struct DT {
        // Reactant species
        using reactants = pmacc::mp_list<Pair<PIC_Tritons, PIC_Deuterons>>;
        // Filter for reactants
        using FilterPair = OneFilter<filter::All>;
        
        // Product species (multiplicities deduced automatically)
        using products = pmacc::mp_list<Pair<PIC_Neutrons, PIC_He4>>;

        // Cross-section parameters (Bosch–Hale parameterization)
        struct Params {
            static constexpr float_X BG = 34.3827_X;
            static constexpr float_X A1 = 6.927e4_X;
            static constexpr float_X A2 = 7.454e8_X;
            // ... additional parameters
        };
        
        using CrossSectionInterpolator = relativistic::FusionFunctor<Params>;
    };

.. note::
   Product multiplicities (stoichiometric coefficients) are deduced automatically
   from species charge and mass numbers. You only declare the product species;
   multiplicity caps are inferred at compile time.

   Examples:

   - T + T → ⁴He + 2n:

     .. code-block:: cpp

         using products = pmacc::mp_list<Pair<PIC_Neutrons, PIC_He4>>;

   - B + p → 3×⁴He:

     .. code-block:: cpp

         using products = pmacc::mp_list<Pair<PIC_He4, PIC_He4>>;

Complete Example: D-T Fusion Reaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a complete example for the Deuterium-Tritium fusion reaction (D + T → n + He⁴).
This is the easiest fusion reaction to achieve and is commonly used in fusion energy research.

The cross-section parameters are based on the Bosch-Hale parameterization from experimental data
(Source: https://hdl.handle.net/11858/00-001M-0000-0027-6535-1, page 29).

.. code-block:: cpp

    struct DT
    {
        //! Reactant species: Deuterium and Tritium
        using reactants = pmacc::mp_list<Pair<PIC_Tritons, PIC_Deuterons>>;
        
        //! Filter for reactants (mandatory for ColliderFromStruct)
        //! Use filter::All to include all particles, or define custom filters
        using FilterPair = OneFilter<filter::All>;
        
        //! Product species: Neutron and Helium-4
        using products = pmacc::mp_list<Pair<PIC_Neutrons, PIC_He4>>;
        
        /** Cross-section parameters for D-T fusion */
        struct Params
        {
            //! Gamow constant: BG = π·α·Z₁·Z₂·√(2·μ·c²) in keV^(1/2)
            //! where α is the fine structure constant, Z₁,Z₂ are charges, μ is reduced mass
            static constexpr float_X BG = 34.3827_X;
            
            //! Bosch-Hale parameterization coefficients for D-T cross-section
            //! These coefficients fit experimental cross-section data
            static constexpr float_X A1 = 6.927e4_X;
            static constexpr float_X A2 = 7.454e8_X;
            static constexpr float_X A3 = 2.050e6_X;
            static constexpr float_X A4 = 5.2002e4_X;
            static constexpr float_X A5 = 0.0_X;
            
            //! Additional Bosch-Hale coefficients
            static constexpr float_X B1 = 6.38e1_X;
            static constexpr float_X B2 = -9.95e-1_X;
            static constexpr float_X B3 = 6.981e-5_X;
            static constexpr float_X B4 = 1.728e-4_X;
        };
        
        //! Cross-section calculation using Bosch-Hale parameterization
        //! Returns cross-section in milli-barns
        using CrossSectionInterpolator = relativistic::FusionFunctor<Params>;
    };

2) Configure the fusion pipeline

.. code-block:: cpp

    // Single reaction
    using FusionPipeline = pmacc::mp_list<ColliderFromStruct<DT>>;
    
    // Multiple reactions (processed sequentially)
    using FusionPipeline = pmacc::mp_list<
        ColliderFromStruct<DT>,
        ColliderFromStruct<DD_branch1>,
        ColliderFromStruct<DD_branch2>
    >;

.. note:: Alternative explicit form

   You can also define the fusion pipeline using the explicit ``Collider`` template:

   .. code-block:: cpp

      using FusionPipeline = pmacc::mp_list<
          Collider<DT_He4n::CrossSectionInterpolator, DT_He4n::reactants, DT_He4n::products, OneFilter<filter::All>>,
          Collider<DD_Tp::CrossSectionInterpolator, DD_Tp::reactants, DD_Tp::products, OneFilter<filter::All>>,
          Collider<DD_He3n::CrossSectionInterpolator, DD_He3n::reactants, DD_He3n::products, OneFilter<filter::All>>,
          Collider<He3D_He4p::CrossSectionInterpolator, He3D_He4p::reactants, He3D_He4p::products, OneFilter<filter::All>>,
          Collider<He3T_He4D::CrossSectionInterpolator, He3T_He4D::reactants, He3T_He4D::products, OneFilter<filter::All>>,
          Collider<TT_He4nn::CrossSectionInterpolator, TT_He4nn::reactants, TT_He4nn::products, OneFilter<filter::All>>,
          Collider<Bp_He4He4::CrossSectionInterpolator, Bp_He4He4::reactants, Bp_He4He4::products, OneFilter<filter::All>>
          >;

Simulation Parameters
---------------------

.. code-block:: cpp

    // Memory allocation control
    constexpr uint32_t cellListChunkSize = TYPICAL_PARTICLES_PER_CELL;
    
    // Product weighting thresholds
    constexpr float_X productMinWeighting = 16.1;
    constexpr uint32_t maxFmult = 1e6;
    
    // Debug flags
    constexpr bool debugFusion = false;
    constexpr bool alwaysFuseQ = false;  // Force 100% fusion probability

.. note:: Fusion production multiplier (Fmult)

   Increasing ``maxFmult`` allows more, lighter products to be created, down to
   the ``productMinWeighting`` threshold. Decreasing ``maxFmult`` results in fewer,
   heavier products. ``Fmult`` controls how the consumed reactant weight (the
   minimum of the two reactant weights in a pair) is split into multiple product
   macro-particles while keeping total weight conserved.

   Algorithm (per fused pair):

   - Start with ``Fmult = maxFmult``
   - Compute ``productWeighting = minWeighting / Fmult``
   - If ``productWeighting < productMinWeighting``, reduce ``Fmult`` to
     ``max(1, minWeighting / productMinWeighting)`` and recompute ``productWeighting``

   The base ``productWeighting`` is then distributed across the two reactant
   sites and product species using the site fractions ``W₁…W₄``.

See also
--------
- Physics and algorithms: :doc:`../../models/fusion`
- Related: :doc:`../../models/binary_collisions`
