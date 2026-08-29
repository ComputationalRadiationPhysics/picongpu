.. _fusionReactions:

Fusion Workflow
===============

This page explains how to configure and run the Fusion Extension in simulations.
For the physics background, algorithms, and references, see the model page:
:doc:`../../models/fusion`.

Species Definition Requirements
-------------------------------
All species participating in fusion reactions must be defined with ``massRatio`` and ``chargeRatio`` flags. These are mandatory for the fusion algorithms to function correctly.

To simplify defining particle masses relative to the electron mass, you can define an atomic mass unit (``amu``) constant. This constant is the ratio of one atomic mass unit to the electron rest mass:

.. literalinclude:: ../../../../share/picongpu/tests/Fusion/include/picongpu/param/speciesDefinition.param
   :language: cpp
   :start-after: doc-include-start: amu-definition
   :end-before: doc-include-end: amu-definition
   :dedent:

Example species definition for deuterons:

.. literalinclude:: ../../../../share/picongpu/tests/Fusion/include/picongpu/param/speciesDefinition.param
   :language: cpp
   :start-after: doc-include-start: deuteron-definition
   :end-before: doc-include-end: deuteron-definition
   :dedent:

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

.. literalinclude:: ../../../../share/picongpu/tests/Fusion/include/picongpu/param/fusion.param
   :language: cpp
   :start-after: doc-include-start: DT-reaction
   :end-before: doc-include-end: DT-reaction
   :dedent:

2) Configure the fusion pipeline

.. literalinclude:: ../../../../share/picongpu/tests/Fusion/include/picongpu/param/fusion.param
   :language: cpp
   :start-after: doc-include-start: fusion-pipeline
   :end-before: doc-include-end: fusion-pipeline
   :dedent:

.. note:: Multiple reactions

   You can add multiple reactions to the pipeline by defining additional reaction structs
   and adding them to the ``FusionPipeline`` list. The reactions are processed sequentially
   in the order they appear.

Simulation Parameters
---------------------

.. literalinclude:: ../../../../share/picongpu/tests/Fusion/include/picongpu/param/fusion.param
   :language: cpp
   :start-after: doc-include-start: simulation-parameters
   :end-before: doc-include-end: simulation-parameters
   :dedent:

.. note:: Fusion production multiplier (Fmult)

   The ``Fmult`` parameter controls how the consumed reactant weight (the
   minimum of the two reactant weights in a pair) is split into multiple product
   macro-particles while keeping total weight conserved. Increasing ``maxFmult`` 
   allows more, lighter products to be created. Decreasing ``maxFmult`` results 
   in fewer, heavier products.

   Algorithm (per fused pair):

   1. Start with ``Fmult = maxFmult``
   2. Calculate the base fusion probability ``P``
   3. Adjust ``Fmult`` to ensure valid Monte Carlo sampling:

      - If ``P > 1.0``: Set ``Fmult = 1.0`` and ``P = 1.0`` (a warning is issued)
      - If ``0 < P ≤ 1.0``: Calculate the maximum allowed multiplier as 
        ``Fmult = min(maxFmult, 0.99/P)`` to ensure ``P * Fmult ≤ 0.99``

   4. Update the fusion probability: ``P = P * Fmult``
   5. Compute ``productWeighting = minWeighting / Fmult``
   6. Distribute the ``productWeighting`` across the two reactant sites and 
      product species using the site fractions ``W₁…W₄``

   This approach ensures that the fusion probability remains below 1.0 (required
   for proper Monte Carlo sampling) while maximizing the number of product 
   particles created per fusion event.

See also
--------
- Physics and algorithms: :doc:`../../models/fusion`
- Related: :doc:`../../models/binary_collisions`
