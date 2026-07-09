************************
Defining Your Simulation
************************

...

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
