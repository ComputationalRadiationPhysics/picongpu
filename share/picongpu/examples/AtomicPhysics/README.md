Atomic Physics
==============

.. sectionauthor:: Brian Marre, Tapish Narwal

This example demonstrates the use of the atomic physics[AtomicPhysics vs Ionization]_ and is intended as a starting point for simulations using atomicPhysics(FLYonPIC).

This setup models a homogeneous charge neutral copper plasma with an ion density of 2e28 1/m^3 and an initial copper charge state of 2.

Copper ions are initialized cold, with no initial momentum, in the ground state, while electrons are given an initial Maxwellian-Temperature of 1 keV.

.. warning::

     This setup does not include elastic collisions

The copper atomic state distribution's initial state will relax over time towards equilibrium.

.. warning::

     The exact equilibrium is dependent on the atomic physics model.

     Therefore direct comparisons of different atomic models should not rely on the equilibrium model only.
     This is especially true if comparing to SCFLY/FlyCHK since they include recombination paths which are not yet implemented in atomicPhysics(FLYonPIC)

Atomic Input Data
-----------------

Any simulation using atomicPhysics(FLYonPIC) requires a user provided set of atomic input data files, these files are not provided with PIConGPU due to licensing issues.

This simulation is intended to use the atomic input data files from `https://github.com/ComputationalRadiationPhysics/FLYonPICAtomicTestData`.
If you have access to this repository, execute `./etc/getAtomicInputData.sh` to copy a set of atomic input files to the `atomicInputData`-directory before running the simulation.

.. note::

     compilation alone does not require atomic input data files

Running CI runtime test for atomicPhysics
-----------------------------------------

This setup is also a run-time integration tests of FLYonPIC.

To build the run-time test use the following steps:
1.) copy the atomic physics example using `pic-create` to a new setup folder
2.) go inside the new setup folder
2.) provide the atomic input data using `./etc/getAtomicInputData.sh`, requires access to the atomicData repository.
3.) build the setup using either `./etc/build.sh` or by hand using `pic-build -c "-DPARAM_FORCE_CONSTANT_ELECTRON_TEMPERATURE=true"`.

To run the test supplied with this use one of the three ways described below.
 - run locally
 - run using tbg
 - run using the JUBE CI workflow

Run Locally
___________

From the root directory
1.) run picongpu by hand using `./bin/picongpu -g 1 1 1 -d 1 1 1 -s 25 --periodic 1 1 1`
2.) run `./validate/validateLocally.sh` to validate the simulation results

Run Using TBG
_____________

The setup comes with `.cfg`-files designed to run on NVIDIA A100 (40 GB) GPU nodes from JURECA using the JUBE CI.
To run and validate the result, use the custom `submitAction.sh` together with one of the provided with the `.cfg`-files, or copy the validation folder to the location of the results manually.

Finally the user must execute validate.sh. The user should be careful to set the correct `cfg` and `submitAction` paths.

Run using JUBE CI
_________________

JUBE calls `tbg` for the scaling test we want to perform (weak/strong, different node sizes).

Each scaling test has a set of `.cfg`-files associated. JUBE runs the `cfg`s one by one.

Using a slightly modified submitAction.sh, the `tbg` call copies over the validation folder to where the results are generated.
In the `TPL` file, after the srun to start the simulation, we call the validation script.

Notes
=====

.. [AtomicPhysics vs Ionization]

     excitation states in addition to charge states and de-/excitation and ionization from and to excited states and ground states.
