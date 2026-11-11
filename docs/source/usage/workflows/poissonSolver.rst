.. _usage-workflows-poissonSolver:

Initial fields form particles using Poisson Solver
=================================================

.. sectionauthor:: Edgar Marquardt

If you don't want to assume quasi-neutrality, you can use the Poisson solver to initialize the fields from the charge density of the particles. To do this, add the :ref:`poissonSolver.param <usage-params-extensions>` to your param files and set the parameter ``useSolver`` to ``true``. Using a preconditioned Bi-CGSTAB Poisson solver (see https://arxiv.org/pdf/2503.08935 for a more detailed explanation), it adds a stage to the 0th timestep of the simulation that computes the initial fields from the charge density of the particles. It is important to use boundary conditions that match the physical situation of your simulation.

Configuration
-------------
The Poisson solver is configured through the :ref:`poissonSolver.param <usage-params-extensions>` parameter file. The following parameters are available:
``useSolver``
   Set to ``true`` to use the Poisson solver to initialize the fields from the charge density of the particles.
``maxSolverSteps``
   Maximum number of Bi-CGSTAB iterations to perform. The solver will stop if the residual is below the ``tolerance``, or if the maximum number of iterations is reached.
``tolerance``
   The solver will stop if the L2 norm of the residual is below this value.
``disablePreconditioner``
   Set to ``true`` to disable the preconditioner. This is not recommended, as it will make the solver much slower.
``maxPreconditionerSteps``
   Maximum number of iterations for the Chebyshev preconditioner.
``BoundaryConditions``
   The boundary condition to use for the Poisson solver. See below.

Boundary Conditions
-------------------
The Poisson solver supports the following boundary conditions. The selected condition is applied to all non-periodic faces. Periodic faces are still periodic.
Available options are:
 - ``BoundaryConditionsHomogenousDirichlet``: zero potential on the boundary.
   Recommended for situations with high total charge and much space around it.
 - ``BoundaryConditionsDirichlet<BoundaryFunction>``: prescribed boundary potential from a
   user-defined function.
   Recommended for situations where the boundary potential can be
   estimated beforehand.
   Implement the BoundaryFunction to define the boundary potential.
 - ``BoundaryConditionsHomogenousNeumann``: zero normal derivative on the
   boundary.
   Warning: This can only be used if the total charge is close to zero,
   otherwise it will not lead to correct results.
   Recommended for close to charge neutral starting plasmas.
 - ``BoundaryConditionsNeumann<BoundaryFunction>``: prescribed normal derivative from a
   user-defined function.
   Recommended for cases where the normal electric field at the boundary
   can be estimated beforehand.
   Warning: This requires that the total charge in the volume is equal or
   close to the sum over the values the BoundaryFunction provides.
   Implement the BoundaryFunction to define the outward normal derivative of
   the potential at the boundary.

Warning: If only periodic faces are used the total charge needs to be close to
or equal to zero. Include at least one dirichlet boundary to calculate the field
for strongly charged starting conditions.

If a function-based boundary condition is used:
The user-defined function that defines the boundary potential or normal derivative. Example:
.. code_block:: cpp
    struct BoundaryFunction
    {
        HDINLINE auto operator()(
            math::Vector<double, simDim> const& totalCellCoordinate,
            DataSpace<simDim> const& outwardNormal) const
        {
            return 0.0;
        }
    };
The function takes the coordinate of the cell at the boundary and the outward normal vector as input and returns the potential or normal derivative at that point.
