Defining Your Simulation
========================

Our frontend implements the `PICMI standard <https://picmi-standard.github.io/>`__.
This is a declarative Python interface for particle-in-cell simulation codes
standardized in the community.
We comply with the standard in the following sense:

  * If a feature of PIConGPU can be expressed in terms of the elements defined in the standard,
    we strive to do so.
  * If a feature of PIConGPU cannot be expressed as such,
    we provide an interoperable extension.
    If this feature could be of general interest,
    we strive to feed it back into the PICMI standard.
  * If the PICMI standard contains elements that are not supported by PIConGPU,
    we strive to provide a clear error message.

Very generally, a PICMI input file is a Python script
that defines and uses one or more ``Simulation`` objects.
A minimal PICMI input file is thus:

.. literalinclude:: ../snippets/defining_simulation/minimal_example.py
   :language: python
   :start-after: BEGIN-MINIMAL-EXAMPLE
   :end-before: END-MINIMAL-EXAMPLE

This short script defines a ``Simulation`` instance
with a fixed number of simulations steps to run.
The only other necessary piece of information is
the electromagnetic field solver which -- in turn -- contains information about the grid.
We will see more elements to add to a ``Simulation`` further below.

Input files can carry `PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__ at the top
(all snippets in the repository carry it, too).
Tools like `uv <https://docs.astral.sh/uv/>`__ and `pipx <https://pipx.pypa.io/>`__ and others can use this to install necessary dependencies on-the-fly.
We recommend this approach to fix the version of PIConGPU you are running in your script.
In order to do so, replace the ``@dev`` with a concrete ``@<commit hash>``.
This will make your input file reproducible and clearly document the version to everyone encountering it.
See :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>` for more details on actually running your script.

The PICMI standard defines various methods to interact with a ``Simulation`` instance.
The most useful for interacting with PIConGPU are:

``simulation.run()``
  Generates the PIConGPU input files,
  compiles a tailored binary and submits the simulation
  to the system given by your runtime configuration (e.g. a batch system).
  This is convenient in most scenarios.
  Note that the call returns after the submission
  (see :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
  for the details of what happens under the hood).

``simulation.write_input_file()``
  Only generate the PIConGPU input files.
  This can be useful in more complex workflows and/or for fine-grained control and debugging.

For other means of interacting with your simulation, see the corresponding :ref:`API Documentation <python_package/api/index:API Documentation>`.

The following sections describe the core building blocks
that you will combine in your input files:
grids and solvers, lasers, species, particle distributions and layouts.
The `tutorial`_ below then puts them together in a complete example.

Grids and Solvers
-----------------

The ``Cartesian3DGrid`` defines the spatial domain of your simulation:

* ``number_of_cells``: the number of cells per dimension,
* ``lower_bound`` / ``upper_bound``: the extent of the domain in metres
  (the lower bound must be ``[0.0, 0.0, 0.0]``), and
* ``lower_boundary_conditions`` / ``upper_boundary_conditions``:
  ``"open"`` (absorbing) or ``"periodic"`` per dimension;
  the lower and the upper condition of a dimension must be equal.

The cell size per dimension is derived as ``(upper_bound - lower_bound) / number_of_cells``.

 By default, the simulation is placed on a single GPU.
 To distribute it over several GPUs,
 give the grid the ``picongpu_n_gpus`` parameter (a list):
 ``[N]`` distributes over ``N`` GPUs in the (preferred) ``y`` direction,
 ``[Nx, Ny, Nz]`` over all three directions.
 Optionally, ``picongpu_grid_dist``
 (a per-dimension list of the cell counts assigned to each GPU)
 assigns explicit numbers of cells to the GPUs
 instead of a uniform distribution.
 The grid must be divisible by the GPU count and the super-cell size
 (``picongpu_super_cell_size``, default ``(8, 8, 4)``) in each dimension;
 the frontend checks this for you and tells you which dimension fails.

The time step of the simulation is fixed by one of the two equivalent quantities:

* ``simulation.time_step_size`` in seconds, or
* ``solver.cfl``, the Courant number,

if the solver is ``"Yee"`` (or ``"Lehe"``) on a Cartesian grid:
giving one derives the other from the cell size,
and giving both must yield consistent values
(the frontend checks this and reports a mismatch).
One of the two must be given;
if neither is set, the input file generation fails.

Laser Pulses
------------

Lasers are added to the simulation via the ``picongpu_lasers`` parameter
or the ``add_laser`` method.
All lasers share a few properties and constraints:

* ``wavelength`` in metres,
* ``duration`` in seconds (the 1-sigma width of the intensity profile),
* ``propagation_direction`` and ``polarization_direction``:
  normalized 3D vectors
  (the propagation direction must point into the simulation box,
  i.e. have a positive ``y`` component),
* ``centroid_position``: the position of the pulse at time zero;
  it must be outside of the simulation box,
  such that the pulse enters the box during the simulation.

``GaussianLaser``
  A Gaussian pulse with the parameters
  ``waist`` (the 1/e² radius at focus), ``focal_position``,
  ``phi0`` (the carrier-envelope phase) and the field amplitude.
  The amplitude is given by exactly one of
  ``a0`` (the normalized vector potential) or ``E0`` (the peak electric field in V/m);
  the other is derived.
  By default, the polarization is linear;
  circular polarization is selected via
  ``picongpu_polarization_type=picmi.lasers.PolarizationType.CIRCULAR``.
  Structured beams can be described with the (matching-length) arrays
  ``picongpu_laguerre_modes`` and ``picongpu_laguerre_phases``.

``DispersivePulseLaser``
  A Gaussian pulse with additional dispersion parameters:
  ``picongpu_spectral_support`` (width of the spectral support),
  ``picongpu_sd_si`` (spatial dispersion), ``picongpu_ad_si`` (angular dispersion),
  ``picongpu_gdd_si`` (group delay dispersion) and ``picongpu_tod_si`` (third-order dispersion).

``FromOpenPMDPulseLaser``
  A pulse imported from an `openPMD <https://www.openpmd.org/>`__ file
  (``file_path``, ``iteration``, ``dataset_name``, ...),
  for initial conditions that are too complex to describe analytically.

.. literalinclude:: ../snippets/defining_simulation/laser_variants.py
   :language: python

.. note::

   The ``PlaneWaveLaser`` and ``TWTSLaser`` classes exist
   but currently do not work:
   generating the input files from a simulation that uses one of them fails.
   They are therefore not described in detail here.

Species
-------

A ``Species`` describes one type of particle in your simulation.
Its most important parameters are:

* ``name``:
  the name of the species (also used in the output files).
  If not given, the name is derived from the particle type.
* ``particle_type``:
  the physical identity of the particle.
  This is either an element symbol (``"H"``, ``"He"``, ``"C"``, ...),
  one of the predefined particle types
  (``"electron"``, ``"positron"``, ``"proton"``, ``"anti-proton"``, ``"photon"``, ...)
  or a custom particle of the form ``"other:<name>"``.
  The mass and charge of known particle types are filled in automatically.
* ``charge_state``:
  the initial charge state of an ion
  (0 for neutral, 1 for singly ionized, ...).
  Only meaningful together with an element ``particle_type``.
  Ions that can be ionized further during the simulation
  (see :ref:`Interactions <python_package/selected_topics/interactions:Interactions>`)
  must specify their initial charge state explicitly.
* ``picongpu_fixed_charge``:
  for ion species that are *not* subject to ionization,
  this fixes the charge of all their particles for the entire simulation.
  It can be combined with ``charge_state`` to choose the charge.
* ``mass`` / ``charge``:
  override the (element-)derived mass and charge in SI units.
  This is how you define custom particles (``particle_type="other:my_particle"``).
* ``particle_shape``:
  the particle shape used for current/charge deposition
  (default is quadratic, i.e. TSC).
* ``method``:
  the particle pusher (default is ``Boris``;
  ``Vay`` and ``HigueraCary`` are relativistic variants,
  ``ReducedLandauLifshitz`` adds radiation reaction).

.. _distributions:

Particle Distributions
----------------------

A distribution describes *where* the particles of a species are placed
(their density profile)
and *how they move* initially.
All distributions take

* ``rms_velocity``:
  a 3D vector of thermal velocity spreads in m/s
  (they are converted to a temperature internally), and
* ``directed_velocity``:
  a 3D vector of a collective drift velocity in m/s.

The available distributions are:

``UniformDistribution``
   A constant density throughout the box (``density`` in m⁻³).

   .. note::

      The ``lower_bound``/``upper_bound`` and ``fill_in`` parameters
      are accepted but currently ignored (they log a warning when set
      to non-default values): the density fills the entire simulation
      box. For sub-volume densities use ``AnalyticDistribution``,
      ``GaussianDistribution`` or ``FoilDistribution`` instead.

``GaussianDistribution``
  A constant-density region with Gaussian ramps at the front and the rear
  of the box (in ``y`` direction):
  ``center_front``/``center_rear`` and ``sigma_front``/``sigma_rear``
  give the position and width of the ramps,
  ``power`` the exponent (2 is Gaussian, 4 and up super-Gaussian),
  ``factor`` the (negative) scaling of the ramps,
  and ``vacuum_front`` the vacuum in front of the profile.

``FoilDistribution``
  A thin foil of constant ``thickness`` at position ``front``
  (perpendicular to ``y``),
  with optional exponential pre- and post-plasma ramps
  (``exponential_pre_plasma_length``/``_cutoff`` and ``exponential_post_plasma_length``/``_cutoff``).

``CylindricalDistribution``
  A cylinder of ``radius`` around the axis ``cylinder_axis``
  through the point ``center_position``,
  with an optional exponential pre-plasma ramp
  (``exponential_pre_plasma_length``/``_cutoff``).

``AnalyticDistribution``
  A density given by an analytic expression in the coordinates ``x``, ``y``, ``z``
  (in SI units), written with `sympy <https://www.sympy.org/>`__.
  The expression is compiled into the simulation binary,
  so it is evaluated on the GPU at runtime.

Several species can share the same distribution:
they are then placed at the same positions,
which is the standard way to build charge-neutral plasmas.
The ``density_scale`` parameter of a species
rescales its density relative to the shared profile
(1.0 keeps it unchanged),
and ``simulation.picongpu_base_density``
(default ``1.0e25`` m⁻³) is the reference density
used to normalize the code units.

Layouts
-------

The layout determines the positions of the particles *within* a cell.
It is given per species via ``simulation.add_species(species, layout)``:

``PseudoRandomLayout``
  ``n_macroparticles_per_cell`` particles per cell at pseudo-random positions.
  This is the default choice for most simulations.

``GriddedLayout``
  A regular sub-grid of ``n_macroparticles_per_cell = [nx, ny, nz]``
  positions per cell (``nx * ny * nz`` particles per cell).
  Useful for well-resolved, low-noise configurations.

``OnePositionLayout``
  A single position per cell
  (``n_macroparticles_per_cell`` particles per cell, all at the same point,
  shifted by ``in_cell_offset`` in units of the cell size).

.. literalinclude:: ../snippets/defining_simulation/warm_plasma.py
   :language: python

The above snippet builds a warm, quasi-neutral plasma:
ions and electrons share the same uniform density profile
(and thus the same particle positions),
each cell carries 8 macroparticles on a 2×2×2 sub-grid.

.. _tutorial:

Tutorial: Setting up a simple LWFA
----------------------------------

We will now add some interesting physics to our minimal example.
This tutorial is supposed to give you a good introduction to the features
you will typically use in your daily work.
More details can be found in :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.

Extracting global constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For starters, it is typically helpful to have access to some parameters in different parts of your input.
In order to do so, we extract some constants and decompose the definition of the solver:

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :end-before: END-LWFA-CONSTANTS

Lasers
^^^^^^

There are various lasers defined in `the PICMI standard <https://picmi-standard.github.io/>`__ and its :ref:`PIConGPU extension <PICMI_Extensions>`.
We define a Gaussian laser as moving into positive ``y`` direction
(this is the convention PIConGPU is optimized for):

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-LASER
   :end-before: END-LWFA-LASER

Species and particles
^^^^^^^^^^^^^^^^^^^^^

In the PICMI standard we define `abstract species <https://picmi-standard.github.io/>`__
and `distributions <https://picmi-standard.github.io/>`__ of particles belonging to such species among the cells.
The precise location of a particle inside of a cell is finally determined by `the layout <https://picmi-standard.github.io/>`__.
Thus, in order to add particles to our simulation we need three components:

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-SPECIES
   :end-before: END-LWFA-SPECIES

We add two species:
``hydrogen``, initialized from the ``GaussianDistribution``,
and ``electrons``, which is initially empty (``initial_distribution=None``).
The hydrogen is created in its ground state (``charge_state=0``, i.e. neutral),
so the plasma is charge neutral before ionization sets in.
The electron species does not receive any initial particles;
they are created by the ionization model below.

We can add various `interactions <https://picmi-standard.github.io/>`__ among our species.
As an example, we allow to ionize the hydrogen into the corresponding electron species:

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-ADK
   :end-before: END-LWFA-ADK

Creating the simulation
^^^^^^^^^^^^^^^^^^^^^^^

We can now create the simulation,
passing it the solver, the laser, the ionization interaction and the species:

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-SIMULATION
   :end-before: END-LWFA-SIMULATION

Diagnostics
^^^^^^^^^^^

Diagnostics, i.e. simulation output, are an important part of your simulation.
PIConGPU allows to define general diagnostics in a flexible way.
See :ref:`the diagnostics topic <python_package/selected_topics/index:Selected Topics>` for a full overview of the capabilities.
There are also various predefined diagnostics you can choose from.
Some of these provide quick access to heavily used features/debugging tools.
Others provide some optimized code for the diagnostic.
For example, we add a checkpoint and a macro-particle counter
(a useful tool for debugging the particle content of your simulation):

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-DIAGNOSTICS
   :end-before: END-LWFA-DIAGNOSTICS

Running the simulation
^^^^^^^^^^^^^^^^^^^^^^

As a last step, we add the following lines to run the simulation upon execution of the script:

.. literalinclude:: ../snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-RUN
   :end-before: END-LWFA-RUN

(De-)serializing a Simulation
-----------------------------

The PyPIConGPU middle layer representation of your simulation
and the individual PICMI elements
are based on `Pydantic <https://docs.pydantic.dev/>`__.
This provides automatic validation and (de-)serialization capabilities.
You can serialize the PyPIConGPU representation of your ``Simulation``
into a machine-readable `JSON <https://json.org/>`__ representation
and recover individual elements from such a representation:

.. literalinclude:: ../snippets/defining_simulation/serialize_simulation.py
   :language: python
   :start-after: BEGIN-SERIALIZE-SIMULATION
   :end-before: END-SERIALIZE-SIMULATION

We refer the reader to the `official documentation <https://docs.pydantic.dev/latest/concepts/serialization/>`__ for further details.
Such a JSON representation of the simulation
can be found in ``metadata/pypicongpu_runner.json``
in every generated set of input files.
As such, you can flexibly reuse various aspects of your previous simulations.

Multiple simulations in a single script
---------------------------------------

With Python as a base language,
there's nothing hindering us from defining, manipulating or even running
multiple simulations from within a single script.
While more complex workflows will probably benefit from a full-blown workflow engine,
there are still some interesting applications for this.

As an example application we will consider
an optimization of the focal position in a Laser Wakefield Acceleration (LWFA) simulation.
We can maximize the ejection of electrons from the plasma
in a particular energy range by adjusting the focal position of the laser.
Very loosely speaking:
The earlier we focus, the more energy can be transferred to the plasma.

Wrap ``Simulation`` definition into function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a first step, it typically makes sense
to wrap the definition of our simulation into a tailored interface
exposing only those degrees of freedom we're actually interested in:

.. literalinclude:: ../snippets/defining_simulation/multiple_simulations.py
   :language: python
   :start-after: BEGIN-MS-WRAP
   :end-before: END-MS-WRAP

The ``FIXED_KWARGS`` and ``FIXED_LASER_KWARGS`` are global dictionaries
containing common parameters.
Note that the simulation is equipped with an ``EnergyHistogram`` diagnostic
that we will use for the post-processing below.
We will use ``make_simulation`` as a shortcut to defining many simulations
that are identical up to the laser focal position in propagation direction.

Static parameter scans
^^^^^^^^^^^^^^^^^^^^^^

In order to get a general idea where to look for optimal parameters
we will start with a static parameter scan:

.. literalinclude:: ../snippets/defining_simulation/multiple_simulations.py
   :language: python
   :start-after: BEGIN-MS-SCAN
   :end-before: END-MS-SCAN

Each simulation writes its input files and results
to its own ``scan/focal_<position>/`` directory,
so that the runs do not interfere with each other.

Immediate post-processing
^^^^^^^^^^^^^^^^^^^^^^^^^

Our static parameter scan is now submitted to the cluster
and we have to wait for the simulations run and finish.
If we want to programmatically post-process the results in the same script,
we have to wait until all simulations have run.
As this is very system specific,
we don't provide an officially supported method for doing so.

We can use the output of the ``EnergyHistogram`` diagnostic
that a run directory contains
to post-process the results,
e.g. to count the electrons in a particular energy range (given in keV)
and to plot it:

.. literalinclude:: ../snippets/defining_simulation/postprocess_histogram.py
   :language: python
   :start-after: BEGIN-POSTPROCESS-HISTOGRAM
   :end-before: END-POSTPROCESS-HISTOGRAM

Dynamic parameter scans / optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the above plot, you can easily read off a good estimate for the ``focal_position``.
But we want to do better and run a full optimization on the problem.
In order to do so, we define our target function as follows:
for any given focal position, this defines and runs the simulation,
then reads the results and returns the value of interest.
(As noted above, the wait for the submitted job to finish
is system specific and not provided by the package.)
This function can be used in an optimization routine, for example:

.. literalinclude:: ../snippets/defining_simulation/optimize_focal_position.py
   :language: python
   :start-after: BEGIN-OPTIMIZE-FOCAL-POSITION
   :end-before: END-OPTIMIZE-FOCAL-POSITION
