===========================
Test for the Poisson solver
===========================

This regression test checks the electrostatic Poisson stage used during PIConGPU initialization.
It validates both the basic Gauss-law consistency of the solver and the fully analytic
reference cases for the discrete operator used in the current implementation.

The test is split into two parts:

* ``gauss_law``: a self-consistency check for the open-boundary and periodic setups.
  This verifies that the solver reproduces the expected charge-density and field relations
  for the standard Poisson problem.
* ``analytic_V``: three analytic cases, each designed for a specific boundary-condition
  pattern. These are run as small, deterministic regression tests against the electrostatic
  potential ``FieldV`` and the reconstructed electric field ``E``.

The analytic cases cover:

* A: fully periodic sine mode, corresponding to the homogeneous Neumann case.
* B: grounded box with homogeneous Dirichlet boundaries on all sides.
* C: Dirichlet in the x-direction and periodic boundaries in y and z, corresponding to a
  slab-like density profile.

The test compares the numerical solution against closed-form expressions for the potential,
charge density, and field. In addition, the validator reconstructs the effective injected
density from the discrete operator and checks its sign and relative L2 error before comparing
field values. This catches common mistakes such as sign-flipped species charges or wrong
boundary-condition setups early.

To execute the full test suite, run the top-level driver:

.. code-block:: bash

   ./share/picongpu/tests/PoissonSolver/bin/ci.sh \
       ./share/picongpu/tests/PoissonSolver \
       ./run02

The script creates the input sets, builds the required solver configurations, runs the
Gauss-law checks and the three analytic test cases, and validates the results automatically.
The second argument is the output directory for the generated build tree and simulation results.

A more explicit manual setup is also possible:

.. code-block:: bash

   pic-create ./share/picongpu/tests/PoissonSolver/gauss_law ./build/poisson-gauss
   pic-create ./share/picongpu/tests/PoissonSolver/analytic_V ./build/poisson-analytic-neumann
   pic-create ./share/picongpu/tests/PoissonSolver/analytic_V ./build/poisson-analytic-dirichlet

   cd ./build/poisson-gauss && pic-build
   cd ./build/poisson-analytic-neumann && pic-build
   cd ./build/poisson-analytic-dirichlet && pic-build -c "-DPARAM_OVERWRITES=-DPARAM_POISSON_BOUNDARY_CONDITION_DIRICHLET"

The analytic cases require the corresponding boundary-condition configuration. Their periodicity
is selected through the simulation setup and the validator expects the patterns ``1 1 1``,
``0 0 0``, and ``0 1 1`` for cases A, B, and C, respectively. This test is intended to detect
regressions in the discrete Poisson solve, the source-mean handling for Neumann problems, and
the correct reconstruction of the electrostatic field for the current implementation.
