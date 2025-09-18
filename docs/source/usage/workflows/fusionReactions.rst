.. _fusionReactions:

Fusion Extension
================

Overview
--------

The PIConGPU Fusion Extension enables Monte Carlo simulation of nuclear fusion reactions between macro-particles based on [Wu2021]_ and [Higginson2019]_. The extension implements fully relativistic binary collision algorithms with local charge and (if possible) mass conservation, producing physically accurate fusion products with proper energy-momentum distributions.

**Use Cases:**
- Inertial confinement fusion (ICF) plasma simulations
- High-energy density physics with fusion heating
- Thermonuclear burn studies in laser-plasma interactions

**Key Features:**
- Relativistic collision algorithm and cross-section evaluation
- Stoichiometric multiplicity with compile-time weight validation
- Local charge conservation always; local mass conservation when non-degenerate
- Support for multiple simultaneous fusion channels

How to Use the Fusion Extension
===============================

Species Definition Requirements
-------------------------------

All species participating in fusion reactions must be defined with ``massRatio`` and ``chargeRatio`` flags. These are mandatory for the fusion algorithms to function correctly.

To simplify defining particle masses relative to the electron mass, PIConGPU provides a constant for the atomic mass unit (`amu`). This constant is the ratio of one atomic mass unit to the electron rest mass. You can use it in `speciesDefinition.param` as follows:

**Example species definition:**

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

**Important:** Mass ratios must be as precise as possible since fusion energy release is calculated from the mass deficit between reactants and products.

Fusion Reaction Configuration
-----------------------------

Configure fusion reactions in ``include/picongpu/param/fusion.param``:

**1. Define Reaction Parameters**

.. code-block:: cpp

    struct DT {
        // Specify reactant species
        using reactants = pmacc::mp_list<Pair<PIC_Tritons, PIC_Deuterons>>;
        // Specify filter for reactants
        using FilterPair = OneFilter<filter::All>;
        
        // Specify product species  
        using products = pmacc::mp_list<Pair<PIC_Neutrons, PIC_He4>>;

        
        // Cross-section parameters (Bosch-Hale parameterization)
        struct Params {
            static constexpr float_X BG = 34.3827_X;    // Gamow constant
            static constexpr float_X A1 = 6.927e4_X;    // Fit coefficients
            static constexpr float_X A2 = 7.454e8_X;
            // ... additional parameters
        };
        
        using CrossSectionInterpolator = relativistic::FusionFunctor<Params>;
    };

.. note::

   Product multiplicities (stoichiometric coefficients) are deduced automatically from species charge and mass numbers. You only need to declare the product species; multiplicity caps are inferred at compile time.

   Examples:

   - T + T → n + ⁴He (also covers T + T → ⁴He + 2n): it suffices to write

     .. code-block:: cpp

         using products = pmacc::mp_list<Pair<PIC_Neutrons, PIC_He4>>;

   - B + p → 3×⁴He: it suffices to write

     .. code-block:: cpp

         using products = pmacc::mp_list<Pair<PIC_He4, PIC_He4>>;

**2. Configure Fusion Pipeline**

.. code-block:: cpp

    // Single reaction
    using FusionPipeline = pmacc::mp_list<ColliderFromStruct<DT>>;
    
    // Multiple reactions (processed sequentially)
    using FusionPipeline = pmacc::mp_list<
        ColliderFromStruct<DT>,
        ColliderFromStruct<DD_branch1>,
        ColliderFromStruct<DD_branch2>
    >;

.. note::
   To use ``ColliderFromStruct<T>`` the reaction struct ``T`` must define these public aliases:
   ``FilterPair``, ``reactants``, ``products``, and ``CrossSectionInterpolator``.

.. note:: Alternative explicit form

   You can also define the fusion pipeline using the explicit ``Collider`` template (without a struct), by passing ``CrossSectionInterpolator``, ``reactants``, ``products``, and a ``Filter`` type directly:

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


**3. Set Simulation Parameters**

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

   ``Fmult`` controls how the consumed reactant weight (the minimum of the two reactant weights in a pair) is split into multiple product macro-particles while keeping total weight conserved.

   Algorithm (per fused pair):

   - Start with ``Fmult = maxFmult``
   - Compute ``productWeighting = minWeighting / Fmult``
   - If ``productWeighting < productMinWeighting``, reduce ``Fmult`` to ``max(1, minWeighting / productMinWeighting)`` and recompute ``productWeighting``
   - Guarantees: ``productWeighting ≥ productMinWeighting`` and ``Fmult ≥ 1`` while conserving the total produced weight ``minWeighting``

   Increasing ``maxFmult`` allows more, lighter products to be created, down to the ``productMinWeighting`` threshold. Decreasing ``maxFmult`` results in fewer, heavier products.
   
   The base ``productWeighting`` is then distributed across the two reactant sites and product species using the site fractions ``W₁…W₄`` (see Particle Creation Algorithm). In symmetric cases this often results in roughly half of a species' weight at each reactant site.

**Parameter Guidelines:**
- ``productMinWeighting``: Minimum allowed weight per two created product macro-particles; 16 let's the particle split in half 4 times before hitting the threshold. (see: ''Fusion Particle Creation Algorithm'')
- ``maxFmult``: Upper bound for how many product macroparticles can be spawned per fused pair; higher values allow more (lighter) products down to ``productMinWeighting``.
- ``cellListChunkSize``: Use ``TYPICAL_PARTICLES_PER_CELL`` for optimal memory usage.

Fusion Particle Creation Algorithm
==================================

Purpose
-------

The fusion particle creation algorithm enforces **local charge conservation** by calculating compile-time weights for product particles. Local charge conservation within each computational cell is essential for numerical stability in PIC simulations, preventing spurious electric fields from charge imbalances.

Problem Definition
------------------

**Input:** Two reactant particles of some weight undergoing fusion.

The Fusion algorithm determines the amount of fuel (reactants) to consume in the fusion process. This fuel has an associated weight W_p that represents the total weight of products (of each species) to be created. The fractional weights W₁, W₂, W₃, W₄ are then multiplied by W_p to determine the actual weights of the outgoing particles.

**Parameter Definitions:**
- q₁ = charge of reactant 1
- q₂ = charge of reactant 2  
- q₃ = charge of product species 1
- q₄ = charge of product species 2
- m₁ = mass of reactant 1
- m₂ = mass of reactant 2
- m₃ = mass of product species 1
- m₄ = mass of product species 2
- W_p = total weight of products (fuel consumption amount)

**Output:** Four product particles with final weights W₁×W_p, W₂×W_p, W₃×W_p, W₄×W_p
- Product species 1: weights W₁×W_p at reactant 1 position, W₃×W_p at reactant 2 position
- Product species 2: weights W₂×W_p at reactant 1 position, W₄×W_p at reactant 2 position

Stoichiometric Multiplicity Limits and Invariants
----------------------------------

We derive compile-time multiplicity limits (c₃, c₄) from species charges (Z) and mass numbers (A). These limits define how much of each product species must be created in total (across both reactant sites).

- Non-degenerate (det ≠ 0), where det := q₃m₄ − q₄m₃:

  .. math::

     \begin{bmatrix} q_3 & q_4 \\ m_3 & m_4 \end{bmatrix}
     \begin{bmatrix} c_3 \\ c_4 \end{bmatrix}
     =
     \begin{bmatrix} q_1{+}q_2 \\ m_1{+}m_2 \end{bmatrix}

- Degenerate (q₃/m₃ = q₄/m₄): symmetric limits

  .. math::

     c_3 = c_4 = \frac{q_1{+}q_2}{q_3{+}q_4}\quad (q_3{+}q_4 \ne 0),\qquad
     c_3 = c_4 = \frac{m_1{+}m_2}{m_3{+}m_4}\quad (\text{if } q_3{+}q_4 \approx 0)

With multiplicity limits, the invariants are:
- Weight bounds: 0 ≤ Wᵢ ≤ cᵢ (i ∈ {product 1, product 2} per site)
- Weight conservation: W₁ + W₃ = c₃, W₂ + W₄ = c₄
- Local charge: q₁ = W₁q₃ + W₂q₄ and q₂ = W₃q₃ + W₄q₄
- Local mass (when Algorithm 1 is valid): m₁ = W₁m₃ + W₂m₄ and m₂ = W₃m₃ + W₄m₄

Algorithm Implementation
-----------------------------

**Algorithm 1: Mass-Charge Conservation under Multiplicity Limits** 
Solve for site-1 weights (W₁, W₂) using:

.. math::

   \begin{bmatrix} q_3 & q_4 \\ m_3 & m_4 \end{bmatrix}
   \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = \begin{bmatrix} q_1 \\ m_1 \end{bmatrix}

Then set W₃ = c₃ − W₁ and W₄ = c₄ − W₂. Accept only if det ≠ 0, all 0 ≤ Wᵢ ≤ cᵢ, and site-2 mass/charge match.

Notes:
- This works even if a product is neutral (det can still be non-zero).
- When valid, both local charge and mass are satisfied.

**Algorithm 2: Charge-Only Conservation under Multiplicity Limits**
If Algorithm 1 is invalid (det ≈ 0 or out-of-limit weights), enforce only local charge with multiplicity limits:
- Always set W₃ = c₃ − W₁ and W₄ = c₄ − W₂.
- Case handling is neutral-friendly and macroparticle-minimizing:
  - Both products neutral: W₁ = W₂ = 0; W₃ = c₃; W₄ = c₄
  - One product neutral: solve on the charged product; put all neutrals at one site
  - Both charged: prefer a single-species split if possible; otherwise bind one limit and solve the other
- Allows W > 1 when multiplicity limits > 1 (e.g., multi-particle channels)

Algorithm 1 is considered failed when:
- det ≈ 0 (identical q/m ratio products), or
- any computed Wᵢ falls outside its multiplicity limit

Implementation Details
---------------------

Both algorithms are evaluated at **compile time** using constexpr functions:

- ``computeStoichiometryCaps()``: derives (c₃, c₄) from species
- ``calculateMassChargeConservingWeightsWithCaps()``: local mass+charge under multiplicity limits
- ``calculateChargeOnlyWithCaps()``: charge-only robust fallback under multiplicity limits

Compile-time validation ensures charge invariants, limit sums, and (when applicable) mass invariants. High-precision tolerances prevent numerical issues.

Code Flow and Implementation
============================

Fusion Extension Architecture
-----------------------------

The fusion extension follows this execution flow::

    simulation.hpp → Fusion.x.cpp → Collider.hpp → WithPeer.hpp → InterCollision.hpp
                                                                 └─ IntraCollision.hpp 

**Core Components:**

- ``Inter/Intra-Collision.hpp``: Combines collision algorithm from the Collision Extension with the Creation Kernel
- ``FusionFunctor.hpp``: Interfaces between collision framework and physics algorithms  
- ``FusionAlgorithm.hpp``: Implements relativistic fusion physics and product momentum calculation

**Algorithm Chain:**

.. code-block:: text

                        uses: (particles/fusion/detail)
    InterCollision.hpp ──────────────────────────────→ FusionFunctor.hpp → FusionAlgorithm.hpp
                     └─ Creation.hpp

Main Physical Algorithms
========================

1. Fusion Process - Binary Particle Collisions
-----------------------------------------------

**Physical Model:** Two reactant macro-particles undergo fusion, producing up to 4 product macro-particles
- Reactant selection within computational cells
- Energy-dependent fusion probability: σ(E) using empirical fits
- Statistical weighting to handle macro-particle representation

2. Particle Creation Algorithm
------------------------------

**Physical Model:** Local charge (and when possible local mass) conservation during particle creation under stoichiometric caps

**Algorithm 1 - Mass-Charge (under caps):**
2×2 local solve for W₁,W₂; set W₃,W₄ from caps; accept if caps and site-2 checks pass

**Algorithm 2 - Charge-Only (under caps):**
Neutral-aware, macro-minimizing case analysis; always respects caps and local charge

**Key Physics:**
- Local charge conservation: q₁ = W₁q₃ + W₂q₄, q₂ = W₃q₃ + W₄q₄
- Cap constraints: 0 ≤ Wᵢ ≤ cᵢ, W₁ + W₃ = c₃, W₂ + W₄ = c₄
- Local mass conservation when Algorithm 1 is valid

3. Relativistic Framework
-------------------------

**Physical Model:** Fully relativistic treatment using Lorentz invariants

**Implementation in ``FusionAlgorithm.hpp``:**

**Mandelstam Variable Approach:**
[Cannoni2016]_
- Uses Lorentz invariant s = E²_total - (pc)² instead of double transformations
- Avoids numerical errors in highly relativistic regime
- Calculates relative velocity through invariant: γᵣ = (s - m₁²c⁴ - m₂²c⁴)/(2m₁m₂c⁴)

**Lorentz Transformations:**
- Lab frame → Center-of-mass frame for isotropic product generation
- CM frame → Lab frame for final product momenta
- Boost formulas: **p**_lab = **p**_cm + [(**V**_cm·**p**_cm)γ_cm/(γ_cm+1) + γ_cm E_cm/c]**V**_cm/c

**Key Physics:**
- Energy-momentum conservation in all reference frames
- Relativistic energy: E = √[(pc)² + (mc²)²]
- Isotropic angular distribution in CM frame

Common Fusion Reaction Examples  
--------------------------------

.. list-table:: Weight Assignments for Common Fusion Reactions (under multiplicity limits)
   :header-rows: 1
   :widths: 25 20 10 10 10 10

   * - Reaction
     - Charges (q₁,q₂,q₃,q₄)
     - W₁
     - W₂  
     - W₃
     - W₄
   * - D + T → ⁴He + n
     - (1,1,2,0)
     - 0.5
     - 0
     - 0.5
     - 1
   * - D + D → T + p
     - (1,1,1,1)
     - 0.5
     - 0.5
     - 0.5
     - 0.5
   * - D + D → ³He + n
     - (1,1,2,0)
     - 0.5
     - 0.5
     - 0.5
     - 0.5
   * - ³He + D → ⁴He + p
     - (2,1,2,1)
     - 0.5
     - 1
     - 0.5
     - 0
   * - ³He + T → ⁴He + D
     - (2,1,2,1)
     - 1
     - 0
     - 0
     - 1
   * - T + T → ⁴He + 2n
     - (1,1,2,0)
     - 0.5
     - 1
     - 0.5
     - 1
   * - ³He + ³He → ⁴He + 2p
     - (2,2,2,1)
     - 0.5
     - 1
     - 0.5
     - 1
   * - p + ⁶Li → ³He + ⁴He
     - (1,3,2,2)
     - 0.5
     - 0
     - 0.5
     - 1
   * - D + ⁶Li → ⁴He + ⁴He
     - (1,3,2,2)
     - 0.5
     - 0
     - 0.5
     - 1
   * - p + ⁷Li → ⁴He + ⁴He
     - (1,3,2,2)
     - 0.5
     - 0
     - 0.5
     - 1
   * - p + ¹¹B → 3×⁴He
     - (1,5,2,2)
     - 0.5
     - 0
     - 1
     - 1.5

**Weight Interpretation:**
- W₁, W₃: weights of product 1 at reactant positions 1, 2
- W₂, W₄: weights of product 2 at reactant positions 1, 2
- Multiplicity limits can be fractional and > 1; weights may exceed 1 accordingly
- If Algorithm 1 is valid, neutrals can split across sites (mass+charge enforced)
- Otherwise, Algorithm 2 minimizes the number of neutral macroparticles (all neutrals at one site)

References
----------

.. [Wu2021]
        D. Wu, Z. M. Sheng, W. Yu, S. Fritzsche, and X. T. He.
        *A pairwise nuclear fusion algorithm for particle-in-cell simulations: Weighted particles at relativistic energies.*
        AIP Advances 11, 075003 (2021).
        https://doi.org/10.1063/5.0051178

.. [Higginson2019]
        D. P. Higginson, A. Link, and A. Schmidt.
        *A pairwise nuclear fusion algorithm for weighted particle-in-cell plasma simulations.*
        Journal of Computational Physics 388, 439–453 (2019).
        https://doi.org/10.1016/j.jcp.2019.03.020

.. [Cannoni2016]
        M. Cannoni.
        *Lorentz invariant relative velocity and relativistic binary collisions.*
        arXiv:1605.00569 [hep-ph] (2016).
        https://arxiv.org/abs/1605.00569v2
