.. _pypicongpu-running:

Running
=======

This document describes the process of executing a simulation after all
parameters have been configured.

   This doument describes the current state of the implementation,
   which does not implement all necessary features for productive HPC use.
   For HPC it is best to just generate input files and ignore running a simulation entirely.

The starting point is a picmi simulation object, e.g. initialized like
that:

.. code:: python

   from picongpu import picmi
   grid = picmi.Cartesian3DGrid(number_of_cells=[192, 2048, 12],
                                lower_bound=[0, 0, 0],
                                upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
                                lower_boundary_conditions=["open", "open", "periodic"],
                                upper_boundary_conditions=["open", "open", "periodic"])
   solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
   laser = picmi.GaussianLaser(0.8e-6, 5.0e-6 / 1.17741, 5.0e-15,
                               a0=8,
                               propagation_direction=[0, 1, 0],
                               focal_position=[0, 4.62e-5, 0]
                               )
   sim = picmi.Simulation(time_step_size=1.39e-16, max_steps=int(2048), solver=solver)
   sim.add_laser(laser, None)

The object ``sim`` now **bundles all information** and is consequently
the only way we will refer to the parameters.

“Running” a simulation means the execution of the following steps:

1. *Generating* the code (``.param``, ``.cfg``) from the given
   parameters and joining those files with the predifined template
   (``pic-create``)
2. *Building* (compiling) the PIConGPU simulation (``pic-build``)
3. *Running* the simulation (``tbg``) itself

These steps can be invoked using two main methods:

-  the **PICMI native interface** (methods specified by picmi)
-  the **PyPIConGPU Runner**

For basic operation the PICMI native interface is sufficient, but for a
tight control of the individual steps please use the PyPIConGPU runner.

PICMI Native Interface
----------------------

This describes the implementation of the methods to run the simulation
as described by PICMI.

It is mainly supported to conform to PICMI, but PIConGPU is much more
complicated than this interface.

**Unless testing a small scenario, please use the Runner as specified
below.**

Internally, this creates a runner and calls the necessary steps. To
retrieve this runner, call ``sim.picongpu_get_runner()``. (See below for
usage.) **This is not recommended to control the simulation run, because
you will skip all sanity checks.** Instead, create the runner yourself
using the methods outline below.

Only Create Input Data
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # sim init...
   sim.write_input_file("/scratch/mydepartment/myname/mydata/myproject/scenario05")

This creates an input data set to be consumed by PIConGPU – after this
call, PICMI is no longer required for the operation at all. This, in
turn, means that you now have to call the other tools (``pic-build``,
``tbg``) on your own.

The passed location must be a not-yet-existing directory. After this
call this directory will be populated and you can run ``pic-build``
there.

This is intended for environments where you can’t use the Runner as
described below.

Run the Simulation Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # sim init...
   sim.step(sim.max_steps)

This runs the entire build pipeline.

PICMI specifies that you can run an arbitrary number of steps. **This is
not the case for PyPIConGPU.** You can only run **all steps at once**.
(Any other number of steps will raise an error.)

   Note: While it would theoretically be possible to run a PIConGPU
   simulation one step at a time, that would be ludicrously expensive in
   terms of execution overhead an memory throughput. If you want to do
   that, please use PIConGPU without PyPIConGPU.

This also **has no multi-device support**:
Without additional configuration it is impossible to ensure that a
simulation is correctly distributed across multiple nodes.

Unless a scratch directory is configured via the environment variable
``$SCRATCH``, **the results will be saved to the temporary directory**
(``/tmp``). In this case a fat warning will be printed.

To retrieve the result location, use:

.. code:: python

   # sim init...
   sim.step(sim.max_steps)
   runner = sim.picongpu_get_runner()

   import os
   print("results are located at: {}", os.path.join(runner.run_dir, "simOutput"))

Did I emphasize enough that you should rather use the runner?

PyPIConGPU Runner
-----------------

The PyPIConGPU Runner object is the intended way for simulations to be
created and executed.

It is initialized using either **a PICMI simulation object** or
alternatively **a PyPIConGPU simulation object**. (Note: If PICMI is
enough for you, you will probably never see a pypicongpu simulation
object. In this case just forget the last paragraph, just use **your**
simulation object.)

Additionally you can supply up to four locations (see next section), but
all of them are optional. Generally speaking, just setting the
environment variable ``$SCRATCH`` is enough for an okay-ish operation.
(To help your future self please organize yourself a little bit better.)

The Runner object tries to help you by running plenty of checks on your
configuration. If you did an obvious mistake it will instantly fail,
non-obvious mistakes can still cause problems down the road – it is not
uncommon for fails to occur after 2 minutes of building. (So don’t run
away 10 seconds after launch.)

Please **use the constructor for the settings**, as the majority of
checks will occur there. You can access the settings directly (see:
:ref:`pypicongpu_runner_dirty_tricks`), but this will bypass many of the checks and should
normally **only be done for reading** the settings.

Used Locations
~~~~~~~~~~~~~~

-  scratch directory (``scratch_dir``): Directory, where many results
   can be stored. Must be **accessable from all involved machines**.
   (PyPIConGPU does not check that!) If not specified, will be loaded
   from the environment variable ``$SCRATCH``. Possibly just specify
   your scratch location in your environment where you also load
   PIConGPU. Note: Do not use your home as scratch on HPC. Will be left
   empty if not specified. Must already exist if specified.
-  setup directory (``setup_dir``): Holds the parameter set and
   ``pic-build`` is called from here. Will be generated in temporary
   directory (``/tmp``) if not given. Must not yet exist when invoking
   the runner. If you just want to run a simulation, the setup dir is
   not important to you. Forget this paragraph.
-  run directory (``run_dir``): Holds all data associated with a single
   PIConGPU run. This includes the input data (copy of setup dir) and
   **all results**, typically **inside the subdirectory ``simOutput``**.
   Must be **accessable from all involved machines**. (This is not
   checked automatically!) If not given, a directory inside the scratch
   dir will be generated. Must not yet exist when invoking the runner.
-  template directory (``pypicongpu_template_dir``): **Only needs to be
   specified if things fail.** (Specifically, *generation* fails at
   ``pic-create``). Holds the template inside which the generated code
   is placed.

In summary:

+-----------+------------+----------------+---------------------------+---------------------------+-----------------------------------------------------------------+
| directory | must exist | must not exist | created by                | used by                   | description                                                     |
+===========+============+================+===========================+===========================+=================================================================+
| scratch   | yes        | no             | user                      | PyPIConGPU                | holds many run dirs                                             |
+-----------+------------+----------------+---------------------------+---------------------------+-----------------------------------------------------------------+
| setup     | no         | yes            | generation (`pic-create`) | build (`pic-build`)       | holds one simulation setup ("scenario")                         |
+-----------+------------+----------------+---------------------------+---------------------------+-----------------------------------------------------------------+
| run       | no         | yes            | execution (`tbg`)         | user/analysis scripts     | holds results of a single simulation run                        |
+-----------+------------+----------------+---------------------------+---------------------------+-----------------------------------------------------------------+
| template  | yes        | no             | PyPIConGPU source code    | generation (`pic-create`) | holds predefined project template -- **can usually be ignored** |
+-----------+------------+----------------+---------------------------+---------------------------+-----------------------------------------------------------------+


Normal Operation
~~~~~~~~~~~~~~~~

.. code:: python

   # sim initialized ...
   from picongpu.pypicongpu.runner import Runner
   r = Runner(sim, run_dir="/scratch/mydata/new_run")
   r.generate()
   r.build()
   r.run()

   import os
   results_dir = os.path.join(r.run_dir, "simOutput")
   analyze(results_dir)

Set the parameters (``setup_dir``, ``scratch_dir``, ``run_dir``,
``pypicongpu_template_dir``) in the constructor of the Runner. All
parameters are optional, pypicongpu will try to guess them for you.

Some sanity checks are performed on all given parameters, and all paths
will be translated to absolute paths. For all paths only a small
character set is allowed: Alphanum, ``-_.`` (and ``/``): This is to
ensure compatibility to the used tools, as they frequently have hiccups
on “special” characters. The used dirs will be logged with the level
info.

After the Runner has been constructed the used directories can be
accessed directly:

.. code:: python

   r = Runner(sim, run_dir="/scratch/mydata/new_run")

   # re-do some checks already performed inside the Runner
   import os
   assert not os.path.exists(r.run_dir)
   assert not os.path.exists(r.setup_dir)

   r.generate()
   assert os.path.isdir(r.setup_dir)

   import tempfile
   if r.setup_dir.startswith(tempfile.gettempdir()):
       print("will build inside tempdir")

In addition to the checks in the constructor, some sanity checks are
performed when invoking the individual steps. These steps are (as
outlined above):

-  ``generate()``: translate the PICMI simulation to
   PIConGPU-understandable files
-  ``build()``: prepare the simulation for execution by calling
   ``pic-build``
-  ``run()``: execute the actual simulation

Every step can only be **called once**, and exactly **in that order**.

If a step itself fails (i.e. the invoked programs return something else
than 0), execution is aborted and the program output (stdout & stderr)
printed.

It is completely valid (and intended design goal) that you can only run
some of the steps: When ``run()`` (or more precisely ``tbg``) can’t
handle your local job submission system, you might only build:

.. code:: python

   r = Runner(sim, setup_dir="/scratch/mysetups/setup01")
   r.generate()
   r.build()
   # ... now manually submit the job from /scratch/mysetups/setup01

.. _pypicongpu_runner_dirty_tricks:

Dirty Tricks
~~~~~~~~~~~~

While the Runner is designed to control **the entire lifecycle** of a
simulation, you can try to feed it a simulation/setup where other steps
have been performed externally.

For that **skip the checks in the constructor**: Construct the object
using an empty simulation and then overwrite the generated paths:

.. code:: python

   from picongpu.pypicongpu.runner import Runner
   from picongpu.pypicongpu.simulation import Simulation

   empty_sim = Simulation()
   # leave all dirs empty, this way no checks will trigger
   r = Runner(empty_sim)
   # now overwrite setup dir
   r.setup_dir = "/other/prepared/setup"

   # skip r.generate() (which would fail), directly build
   r.build()
   r.run()

While you can try that for you local setup, **it is not guaranteed to
work**.
