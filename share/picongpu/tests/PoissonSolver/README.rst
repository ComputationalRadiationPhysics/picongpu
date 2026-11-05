===========================
Test for the Poisson solver
===========================

This test is for validating the PoissonSolver in the PIConGPU initialization. The test setup is a thick, charged tube,
for which the electric field can be calculated analytically. The test compares the analytical solution with the numerical solution
from the Poisson solver.

To run this test, one has to execute ci.sh with the location of the input and output directory.

..code-block:: bash
./picongpu/share/picongpu/tests/PoissonSolver/bin/ci.sh picongpu/share/picongpu/tests/PoissonSolver/ ./run02
