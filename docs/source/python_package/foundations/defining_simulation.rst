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

This short script defines a ``Simulation`` instance
with a fixed number of simulations steps to run.
The only other necessary piece of information is
the electromagnetic field solver which -- in turn -- contains information about the grid.
We will see more elements to add to a ``Simulation`` further below.

At the top of the script, you can see `PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__.
Tools like `uv <https://docs.astral.sh/uv/>`__ and `pipx <https://pipx.pypa.io/>`__ and others can use this to install necessary dependencies on-the-fly.
We recommend this approach to fix the version of PIConGPU you are running in your script.
In order to do so, replace the ``@dev`` with a concrete ``@<commit hash>``.
This will make your input file reproducible and clearly document the version to everyone encountering it.
See :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>` for more details on actually running your script.

The PICMI standard defines various methods to interact with a ``Simulation`` instance.
The most useful for interacting with PIConGPU are:

``simulation.run()``
  Generates PIConGPU input files, compiles a tailored binary and runs this all in one go.
  This is convenient in most scenarios.

``simulation.write_input_file()``
  Only generate the PIConGPU input files.
  This can be useful in more complex workflows and/or for fine-grained control and debugging.

For other means of interacting with your simulation, see the corresponding :ref:`API Documentation <python_package/api/index:API Documentation>`.

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

We add two species with a shared distribution:
``hydrogen`` (initially neutral) and ``electrons`` (initially empty).
Initializing both from the same ``GaussianDistribution``
ensures that they are placed consistently (typically at the same positions),
such that the plasma is charge neutral where the ionization model below will act.

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
