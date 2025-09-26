.. _model-fusion:

Fusion
======

Overview
--------

The PIConGPU Fusion Extension enables Monte Carlo simulation of nuclear fusion reactions between macro-particles based on [Wu2021]_ and [Higginson2019]_. The extension implements fully relativistic binary collision algorithms with local charge and (if possible) mass conservation, producing physically accurate fusion products with proper energy-momentum distributions.

See also: the workflow guide in :doc:`../usage/workflows/fusionReactions` for setup and parameter examples.

Use Cases
---------
- Inertial confinement fusion (ICF) plasma simulations
- High-energy density physics with fusion heating
- Thermonuclear burn studies in laser-plasma interactions

Key Features
------------
- Relativistic collision algorithm and cross-section evaluation
- Stoichiometric multiplicity with compile-time weight validation
- Local charge conservation always; local mass conservation when non-degenerate
- Support for multiple simultaneous fusion channels

Species Definition Requirements
-------------------------------
For species setup details and examples, see the workflow page:
:doc:`../usage/workflows/fusionReactions`.

Fusion Reaction Configuration (Summary)
--------------------------------------
For a minimal configuration pattern and parameter examples, see the workflow page:
:doc:`../usage/workflows/fusionReactions`.

Pipeline configuration and parameters are detailed in the workflow page.

Fusion Particle Creation Algorithm
----------------------------------

Purpose
-------
The fusion particle creation algorithm enforces local charge conservation by calculating compile-time weights for product particles. Local charge conservation within each computational cell is essential for numerical stability in PIC simulations, preventing spurious electric fields from charge imbalances.

Problem Definition
------------------
Input: Two reactant particles of some weight undergoing fusion. The Fusion algorithm determines the amount of fuel (reactants) to consume in the fusion process. This fuel has an associated weight W_p that represents the total weight of products (of each species) to be created. The fractional weights W₁, W₂, W₃, W₄ are then multiplied by W_p to determine the actual weights of the outgoing particles.

Stoichiometric Multiplicity Limits and Invariants
-------------------------------------------------
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
  - Weight bounds: 0 ≤ Wᵢ ≤ cᵢ
  - Weight conservation: W₁ + W₃ = c₃, W₂ + W₄ = c₄
  - Local charge: q₁ = W₁q₃ + W₂q₄ and q₂ = W₃q₃ + W₄q₄
  - Local mass (when Algorithm 1 is valid): m₁ = W₁m₃ + W₂m₄ and m₂ = W₃m₃ + W₄m₄

Algorithm 1: Mass-Charge Conservation under Multiplicity Limits
----------------------------------------------------------------
Solve for site-1 weights (W₁, W₂) using:

.. math::

   \begin{bmatrix} q_3 & q_4 \\ m_3 & m_4 \end{bmatrix}
   \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = \begin{bmatrix} q_1 \\ m_1 \end{bmatrix}

Then set W₃ = c₃ − W₁ and W₄ = c₄ − W₂. Accept only if det ≠ 0, all 0 ≤ Wᵢ ≤ cᵢ, and site-2 mass/charge match.

Algorithm 2: Charge-Only Conservation under Multiplicity Limits
---------------------------------------------------------------
If Algorithm 1 is invalid (det ≈ 0 or out-of-limit weights), enforce only local charge with multiplicity limits. Case handling is neutral-friendly and macroparticle-minimizing.

Implementation Details
---------------------
In  ``include/picongpu/particles/fusion/detail/Creation.hpp`` both algorithms are evaluated at compile time using constexpr functions:

- ``computeStoichiometryCaps()``: derives (c₃, c₄) from species
- ``calculateMassChargeConservingWeightsWithCaps()``: local mass+charge under multiplicity limits
- ``calculateChargeOnlyWithCaps()``: charge-only robust fallback under multiplicity limits

Relativistic Kinematics and Fusion Sampling (Cannoni, 2016)
-----------------------------------------------------------
For each candidate pair we compute kinematics using Lorentz-invariant quantities to avoid loss of precision at high γ, following Cannoni’s formulation of invariant relative velocity [Cannoni2016]_. We adopt the metric signature (+, −, −, −) and four-momentum convention :math:`p_i^\mu = (E_i/c, \vec{p}_i)`.

Key kinematic invariants (computed from lab-frame inputs)

- Total four-momentum: :math:`P^\mu = p_1^\mu + p_2^\mu = (E_{\mathrm{tot}}/c, \vec{p}_{\mathrm{tot}})`.
- Mandelstam s (energy-squared invariant):

  .. math::

     s \;:=\; c^2 P^\mu P_\mu 
       \,=\, E_{\mathrm{tot}}^2 - |\vec{p}_{\mathrm{tot}}|^2 c^2 \,=\, (p_1^\mu + p_2^\mu)^2 c^2.

- Center-of-mass (CM) energy and velocity:

  .. math::

     E_{\mathrm{cm}} = \sqrt{s},\quad
     \vec{V}_{\mathrm{cm}} = \frac{c^2\,\vec{p}_{\mathrm{tot}}}{E_{\mathrm{tot}}},\quad
     \gamma_{\mathrm{cm}} = \left(1- \frac{|\vec{V}_{\mathrm{cm}}|^2}{c^2}\right)^{-1/2}.

- Relative Lorentz factor (Cannoni):

  .. math::

     \gamma_r \,=\, \frac{s - m_1^2 c^4 - m_2^2 c^4}{2 m_1 m_2 c^4}, \qquad
     |v_{\mathrm{rel}}| \,=\, c\,\sqrt{1 - \gamma_r^{-2}}.

Associated Møller (invariant) relative speed (used for rates):

.. math::

   v_{M} \,=\, |v_{\mathrm{rel}}|\,\gamma_{\mathrm{cm}}.

Fusion probability and sampling
-------------------------------
- The microscopic cross section functor :math:`\sigma(E_{\mathrm{cm}})` (e.g., Bosch–Hale) is evaluated at the pair’s CM energy :math:`\sqrt{s}`.
- Over a timestep :math:`\Delta t`, the acceptance probability scales as

  .. math::

     P \;\propto\; \sigma(E_{\mathrm{cm}})\, v_M\, \Delta t

- On acceptance, sample an isotropic direction in the CM frame; compute product total energies

  .. math::

     E_{3,\mathrm{cm}} = \frac{s + (m_3^2 - m_4^2)c^4}{2\,\sqrt{s}}, \qquad
     E_{4,\mathrm{cm}} = E_{\mathrm{cm}} - E_{3,\mathrm{cm}},

and the common CM momentum magnitude from :math:`E^2 = p^2 c^2 + m^2 c^4`. Notation (CM frame). Let :math:`\vec{P}_1` denote the three-momentum of product 1 and :math:`\vec{P}_2` that of product 2, then

.. math::

  \vec{P}_1 = -\vec{P}_2,\qquad |\vec{P}_1| = |\vec{P}_2| = \frac{\sqrt{E_{3,\mathrm{cm}}^2 - m_3^2 c^4}}{c} = \frac{\sqrt{E_{4,\mathrm{cm}}^2 - m_4^2 c^4}}{c}.

Finally, boost both product four-momenta back to the lab frame with :math:`\vec{V}_{\mathrm{cm}}`.

Code Flow and Implementation
----------------------------

Fusion Extension Architecture
-----------------------------

The fusion extension follows this execution flow::

    simulation.hpp → Fusion.x.cpp → Collider.hpp → WithPeer.hpp
                                                                 └─ IntraCollision.hpp 

Core Components
---------------
- ``Inter/Intra-Collision.hpp``: Combines collision algorithm from the Collision Extension with the Creation Kernel
- ``FusionFunctor.hpp``: Interfaces between collision framework and physics algorithms
- ``FusionAlgorithm.hpp``: Implements relativistic fusion physics and product momentum calculation

Algorithm Chain
---------------

.. code-block:: text

                        uses: (particles/fusion/detail)
    InterCollision.hpp ──────────────────────────────→ FusionFunctor.hpp → FusionAlgorithm.hpp
                     └─ Creation.hpp

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

.. [Takizuka1977]
        T. Takizuka and H. Abe.
        *A binary collision model for plasma simulation with a particle code.*
        Journal of Computational Physics 25(3), 205–219 (1977).
        https://doi.org/10.1016/0021-9991(77)90099-7
