************************
Defining Your Simulation
************************

Our frontend implements the `PICMI standard`_.
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
that defines and uses one or more `Simulation`_ objects.
A minimal PICMI input file is thus::

  <see minimal example>

This short script defines a ``Simulation`` instance
with a fixed number of simulations steps to run.
The only other necessary piece of information is
the electromagnetic field solver which -- in turn -- contains information about the grid.
We will see more elements to add to a ``Simulation`` further below.

At the top of the script, you can see `Python script inline metadata`_.
Tools like `uv`_, `pipx`_ and others can use this to install necessary dependencies on-the-fly.
We recommend this approach to fix the version of PIConGPU you are running in your script.
In order to do so, replace the ``@dev`` with a concrete ``@<commit hash>``.
This will make your input file reproducible and clearly document the version to everyone encountering it.
See `Running your simulation`_ for more details on actually running your script.

The PICMI standard defines various methods to interact with a ``Simulation`` instance.
The most useful for interacting with PIConGPU are:

``simulation.run()``
  Generates PIConGPU input files, compiles a tailored binary and runs this all in one go.
  This is convenient in most scenarios.

``simulation.write_input_file()``
  Only generate the PIConGPU input files.
  This can be useful in more complex workflows and/or for fine-grained control and debugging.

For other means of interacting with your simulation, see the corresponding `API documentation`_.

Tutorial: Setting up a simple LWFA
==================================

We will now add some interesting physics to our minimal example.
This tutorial is supposed to give you a good introduction to the features
you will typically use in your daily work.
More details can be found in the various chapters of `the deep dive`_.

Extracting global constants
---------------------------

For starters, it is typically helpful to have access to some parameters in different parts of your input.
In order to do so, we extract some constants and decompose the definition of the solver::

  <NUM_CELLS, CELL_SIZE, grid and solver defined>

Lasers
------

There are various lasers defined in `the PICMI standard`_ and its `PIConGPU extension`_.
We define a Gaussian laser as moving into positive ``y`` direction
(this is the convention PIConGPU is optimized for)::

  <laser definition>

Species and particles
---------------------

In the PICMI standard we define `abstract species`_
and `distribute`_ particles belonging to such species among the cells.
The precise location of a particle inside of a cell is finally determined by the `layout`_.
Thus, in order to add particles to our simulation we need three components::

  <define distribution, layout and MultiSpecies>

A `MultiSpecies`_ consists of multiple individual `Species`_
but ensures that the various species are consistently intialized together
(i.e. typically at the same positions to ensure charge neutrality).

We can add various `interactions`_ among our species.
As an example, we allow to ionize the hydrogen into the corresponding electron species::

  <ADK>

Diagnostics
-----------

Diagnostics, i.e. simulation output, are an important part of your simulation.
PIConGPU allows to define general diagnostics in a flexible way.
See `the corresponding deep dive`_ for a full overview of the capabilities.
There are also various `predefined diagnostics`_ you can choose from.
Some of these provide quick access to heavily used features/debugging tools.
Others provide some optimized code for the diagnostic.
For example, we add a checkpoint and a macro-particle counter
(a useful tool for debugging the particle content of your simulation)::

  <Checkpoint, MacroParticleCount>

Running the simulation
----------------------

As a last step, we add the following lines to run the simulation upon execution of the script::

  <run lines>

(De-)serializing a ``Simulation``
=================================

The PICMI standard is based on `pydantic`_.
This provides automatic validation and (de-)serialization capabilities.
You can serialize your ``Simulation`` into a machine-readable `json`_ representation via::

  simulation.model_dump()

We refer the reader to the `official documentation`_ for further details.
Such a `json`_ representation can be found in ``metadata/picmi_simulation.json``
in every generated set of input files.

A ``Simulation`` can be deserialized from such a representation.
This is particularly useful from a file, e.g.::

  def deserialize_simulation(path):
    with Path(path).open('r') as file:
      return Simulation(json.load(file))

This allows you to recover a PICMI ``Simulation`` instance from the generated input files.

It is also possible to recover individual elements,
if you know what you are looking for.
For example, we could recover the species definitions from a previously run simulation::

  def deserialize_species(path):
    with Path(path).open('r') as file:
      return [Species(spec) for spec in json.load(file)['species']]

As such, you can flexibly reuse various aspects of your previous simulations.

Multiple simulations in a single script
=======================================

With Python as a base language,
there's nothing hindering us from defining, manipulating or even running
multiple simulations from within a single script.
While more complex workflows will probably benefit from a full-blown workflow engine,
there are still some interesting applications for this.

As an example application we will consider
an optimization of the focal position in an `LWFA`_ simulation.
We can maximize the ejection of electrons from the plasma
in a particular energy range by adjusting the focal position of the laser.
Very loosely speaking:
The earlier we focus, the more energy can be transferred to the plasma.
The full example code can be found `here`_.

Wrap ``Simulation`` definition into function
--------------------------------------------

As a first step, it typically makes sense
to wrap the definition of our simulation into a tailored interface
exposing only those degrees of freedom we're actually interested in::

  TRANSVERSE_FOCUS = float(numberCells[0] * cellSize[0] / 2.0)

  def make_laser(focal_position):
    return GaussianLaser(
      **FIXED_LASER_KWARGS,
      focal_position = [TRANSVERSE_FOCUS, focal_position, TRANSVERSE_FOCUS],
    )

  def make_simulation(focal_position):
    return Simulation(
      **FIXED_KWARGS,
      picongpu_laser=[make_laser(focal_position)]
    )

The ``FIXED_KWARGS`` and ``FIXED_LASER_KWARGS`` are global dictionaries
containing common parameters.
We will use ``make_simulation`` as a shortcut to defining many simulations
that are identical up to the laser focal position in propagation direction.

Static parameter scans
----------------------

In order to get a general idea where to look for optimal parameters
we will start with a static parameter scan::

  FOCAL_POSITIONS = []

  simulations = [make_simulation(focal_position) for focal_position in FOCAL_POSITIONS]

  for simulation in simulations:
    simulation.run()

Immediate post-processing
-------------------------

Our static parameter scan is now submitted to the cluster
and we have to wait for the simulations run and finish.
If we want to programmatically post-process the results in the same script,
we have to wait until all simulations have run.
As this is very system specific,
we don't provide an officially supported method for doing so.
But a good-enough solution is defined `in the example script`_ as ``wait_for``::

  for simulation in simulations:
    wait_for(simulation)

We can now use the output of the `EnergyHistogram`_ and plot it::

  def count_electrons_in_energy_range(simulation, min_energy, max_energy):
    counts, bins = EnergyHistogram(energy_histogram_path(simulation)).get(species="electrons", iteration=[FIXED_KWARGS["max_steps"]])[:2]
    return sum(counts[(bins >= min_energy) * (bins < max_energy)])

  plt.plot(FOCAL_POSITIONS, [count_electrons_in_energy_range(simulation, MIN_ENERGY, MAX_ENERGY) for simulation in simulations])
  plt.show()

Dynamic parameter scans / optimization
--------------------------------------

From the above plot, you can easily read off a good estimate for the ``focal_position``.
But we want to do better and run a full optimization on the problem.
In order to do so, we define our target function as follows::

  def electron_count_of(focal_position):
    simulation = make_simulation(focal_position)
    simulation.run()
    wait_for(simulation)
    return count_electrons_in_energy_range(simulation, MIN_ENERGY, MAX_ENERGY)

For any given focal position, this defines and runs the simulation,
then reads the results and returns the value of interest.

This function can be used in an optimization routine, for example::

  from scipy.optimize import minimize

  def maximize(*args, **kwargs):
    result = minimize(lambda *a, **kw: -args[0](*a, **kw), *args[1:], **kwargs)
    result.fun = -result.fun
    return result

  result = maximize(electron_count_of, ESTIMATED_FOCAL_POSITION)
  optimal_focal_position = result.x
  maximal_electron_count = result.fun
