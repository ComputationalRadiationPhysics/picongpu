Quick Start
===========

This quick start guide will get you
from zero to running a complete PIConGPU simulation with minimal effort.
It describes one among many different ways to use PIConGPU.
You can use it as a starting point to explore for yourself
or dive deeper and explore the key concepts and configuration options
with our :ref:`Foundations <python_package/foundations/index:Foundations>` chapter.
If you are more interested in how to setup up a particular aspect of your simulation,
you'll find information in :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.

What You'll Build
-----------------

We will set up and start a minimal simulation
of the electromagnetic field on a small 3D grid
with periodic boundary conditions.
Even though it does not contain any particles or lasers yet,
it exercises the full machinery of the Python package:
input file generation, compilation of a tailored binary and submission of the simulation.
Afterwards, you can add the physical ingredients of your experiment
to the same skeleton (see :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>`).

Prerequisites
-------------

* Python 3.11, 3.12 or 3.13.
* To run locally on your machine: nothing else is required.
* To run on a cluster: access to the cluster
  and, if necessary, an account on the systems that provide the software
  (compilers, libraries) to compile and run PIConGPU.

Step 1: Write Your Input File
-----------------------------

A PIConGPU simulation is defined in a plain Python script.
Create a file ``my_first_simulation.py`` with the following content:

.. literalinclude:: ../snippets/quickstart/my_first_simulation.py
   :language: python

The lines at the top (after the shebang) are `PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__.
Tools like `uv <https://docs.astral.sh/uv/>`__ read them to install the necessary dependencies on-the-fly,
so the script is self-contained and documents the version of PIConGPU it is meant to run with.
To fix the version even more tightly, replace ``@dev`` with a concrete ``@<commit hash>``.

The rest of the script does three things:

* The ``Cartesian3DGrid`` defines the spatial domain:
  128 cells per dimension, each cell 7.8125e-9 m (``upper_bound``/``number_of_cells``) in size,
  with periodic boundary conditions on all sides.
* The ``ElectromagneticSolver`` solves Maxwell's equations with the Yee scheme.
  The parameter ``cfl`` is the Courant number:
  together with the cell size, it determines the size of the simulation time step
  (0.95 is a common, stable choice).
* The ``Simulation`` object ties the solver and the grid together
  and sets the runtime to 100 time steps (``max_steps``).
  Finally, ``simulation.run()`` generates the PIConGPU input files from the simulation,
  compiles a tailored binary and submits the simulation
  (see `Step 3`_ below for what exactly happens).

Step 2: Configure Your Environment
----------------------------------

Besides your simulation, PIConGPU needs a *runtime configuration*:
it describes the system you run on and the metadata (e.g. your name)
to record along with your results.
It is kept in a ``TOML`` file that is found when the ``picongpu`` package is imported
(search order, presets, and all available knobs are documented in
:ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`).

For a local run on your machine, a minimal configuration is sufficient.
Create a file ``.picongpurc.toml`` next to your input script::

  preset = "bash"

The ``preset`` selects the curated environment configuration for your system.
To run on a cluster, use the preset of your system instead
(a full list of the presets shipped with the package is shown in
:ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Presets>`).
Many cluster presets additionally require you to tell PIConGPU who you are,
so that your runs carry proper metadata::

  preset = "rosi-hzdr"
  author = "Your Name"
  email = "you@example.org"

Step 3: Run It
--------------

With ``uv`` installed (see the :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>` page),
the whole thing is a single command:

.. literalinclude:: ../snippets/quickstart/run_with_uv.sh
   :language: bash

``uv`` reads the PEP 723 metadata of the script,
installs the pinned version of PIConGPU (and its dependencies) into an ephemeral environment
and runs the script with it.

If you prefer to install PIConGPU manually into a virtual environment first,
you can also simply execute the script with that environment's ``python``.
The installation is a single ``pip`` command:

.. literalinclude:: ../snippets/quickstart/install_from_git.sh
   :language: bash

When the script runs, the package

#. writes the PIConGPU input files into ``my_first_simulation_setup/``,
#. compiles a PIConGPU binary tailored to exactly this simulation, and
#. submits the simulation to the system given by your runtime configuration:
   on a cluster this is usually a batch system job
   (with the ``bash`` preset, the simulation is started on your machine).

The script returns as soon as the submission is done
and does not wait for the simulation to finish.
On a cluster, you now interact with your job as usual
(e.g. via the job id to monitor its progress).

Step 4: Where Are Your Results?
-------------------------------

After the simulation has run,
the output data is found in the ``simOutput/`` directory of the run.
The run directory (``my_first_simulation_run/``) contains a helper script
``link_results.sh`` that creates a link to that directory,
e.g. into a folder of your choice:

.. literalinclude:: ../snippets/running_simulation/link_results.sh
   :language: bash

For the details on the layout of the run directory,
on the steps the package performs under the hood,
and on how to re-run or inspect individual steps,
see :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`.

Next Steps
----------

* Add physics to your simulation:
  lasers, species, particle distributions and interactions
  are covered in :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>`.
* Record what you want to measure:
  the available diagnostics are documented in :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.
* Take control of your environment:
  presets, fine-tuning and custom profiles are explained in
  :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.
