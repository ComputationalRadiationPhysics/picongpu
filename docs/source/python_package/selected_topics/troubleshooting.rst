Troubleshooting
===============

The common failure modes of the Python interface, grouped by the
*phase* in which they appear, with the message you will see and the
usual fix.
Most of these are configuration problems that you can catch **before**
you compile or submit anything, which is the cheapest place to fix them.

Catch configuration errors early
--------------------------------

After any change to a simulation definition, generate the input files
*without* compiling or submitting:

.. literalinclude:: ../snippets/troubleshooting/validate_before_submit.py
   :language: python
   :start-after: sim = picmi.Simulation

This exercises the same validation and rendering that a real run would,
so a wrong ``cfl``/time step, a bad grid distribution, a missing preset
variable, or a species/layout mismatch all fail here -- seconds instead
of a compile and a queued job.

Input-file generation
---------------------

``delta_t_si ... Input should be a valid number``
   You set neither the solver's ``cfl`` nor the simulation's
   ``time_step_size``.
   Set exactly one of the two (they must agree if you give both).

``GPU- and/or super-cell-distribution in <x|y|z> dimension does not match grid size``
   The number of cells in that dimension is not divisible by the number of
   GPUs times the super-cell size.
   Adjust the grid size, ``picongpu_n_gpus`` or ``picongpu_super_cell_size``
   (a grid is distributed per dimension, see
   :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>`).

``upper and lower boundary conditions must be equal (can only be chosen by axis, not by direction)``
   The lower and upper boundary of one dimension differ.
   PIConGPU chooses the boundary *per axis*, so both ends of a dimension
   must be the same (both ``"open"`` or both ``"periodic"``).

``An initial distribution needs a layout. ...``
   You passed a ``layout`` to a species with ``initial_distribution=None``
   (or the reverse).
   Give a layout only to species that actually get particles,
   and ``None`` to the ones that start empty
   (e.g. an electron species filled by ionization).

``charge_state may only be set for ions``
   You set ``charge_state`` on a predefined particle
   (``"electron"``, ``"proton"``, ...).
   ``charge_state`` is only meaningful for *element* species
   (``"H"``, ``"He"``, ``"C"``, ...).

``setup directory must not exist before generation -- did you call generate() already?``
   The setup directory already exists from a previous call.
   Remove it (or point at a fresh path) before generating again.

Preset and profile
------------------

``MissingVariable: Rendering your profile template encountered a missing variable``
   Your preset's profile template references variables you have not set.
   The message lists them; you can also query
   ``rc_params["required_information"]``.
   Provide the values in your ``picongpurc.toml``
   (or the environment) -- see
   :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.

``Setting preset=... triggered resetting rc_params while it contained non-default content``
   Changing the preset resets the other runtime-configuration parameters
   to the new preset's defaults --
   and with the default ``dirty_reset_policy`` of ``"raise"``
   this raises if you had already customized anything.
   Choose the preset *first*, then set your customizations
   (or set ``dirty_reset_policy`` to ``"warn"`` / ``"ignore"``
   if you need to change the preset later).

Build and submission
--------------------

``No cfg file given (-c|--cfg).`` / ``The given cfg file "..." does not exist (-c|--cfg).``
   The batch configuration file (``N.cfg``) was not found.
   This means the run-time configuration was not fully generated;
   check that the setup directory was created before the build step.

``Destination path already in use, cannot create new folder``
   The destination (run) directory already exists and ``force`` is not set.
   Use a fresh run directory or pass ``force=True`` to
   ``simulation.run()``.

``Possible cyclic dependency detected or unknown variables used!``
   A ``!variable`` in your batch template did not resolve.
   The message lists the offending variable(s);
   define it (via the template or an ``overwrite_vars`` entry)
   or remove the reference.

``Command not found`` / ``No such file or directory`` for ``pic-build``, ``tbg`` or ``pic-create``
   The build/submission tools are not on the ``PATH`` of the workflow,
   usually because the profile (``PIC_PROFILE``) was not loaded
   or points at the wrong PIConGPU source (``PICSRC``).
   Check your runtime configuration and the profile it selects.

Compile / CMake errors during the build step
   The toolchain (compiler, CUDA, MPI, libraries) is not active.
   This is an environment problem, not a simulation-definition problem:
   make sure the profile loads the right modules
   and that ``PICSRC`` / ``PIC_BACKEND`` are correct.

During the run
--------------

``Error: PIConGPU environment profile under "..." not found!``
   The profile is not readable on the *compute* node
   (e.g. it lives on a home directory that is not mounted there).
   Use a profile on a shared filesystem, or load the environment
   some other way on the nodes.

The job finished but I cannot find ``simOutput/``
   The output is in the job's working directory,
   not directly in your run directory.
   Use ``link_results.sh`` (see
   :ref:`HPC Submission Internals <hpc-submission>`)
   to connect it to a location you choose --
   and do it while the (temporary) job directory still exists.

Where to look next
------------------

* :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
  -- the layout of the run directory and the workflow steps.
* :ref:`HPC Submission Internals <hpc-submission>`
  -- what the build and submission steps actually do.
* :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`
  -- presets, profiles and the runtime-configuration parameters.
