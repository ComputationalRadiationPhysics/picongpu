Troubleshooting
===============

.. sectionauthor:: Julian J. Lenz

The common failure modes of the Python interface, grouped by the
*phase* in which they appear, with the message you will see and the
usual fix.
Most of these are configuration problems that you can catch **before**
you compile or submit anything, which is the cheapest place to fix them.

Installation
------------

During installation of the PIConGPU Python package,
package managers **do NOT need to compile anything**.
Users have reported that they try to do so in some environments and typically fail.
This has been traced back to

* either very young versions of Python:
  While PIConGPU itself does not impose restrictions here,
  some of its dependencies apparently lag behind with providing pre-built distributions.
  Explicitly specifying/using a slightly older Python version
  (e.g. ``--python=3.14`` when 3.15 is the latest)
  has been reported to solve this problem.
* or extremely outdated systems:
  When glibc or other fundamental software are ancient,
  package repositories stop providing pre-built distributions for such systems.
  This is clearly a red flag about your system.
  It might be solved by loading some recent compiler modules or the like,
  but you should seriously consider if it's safe to use this system.
  If there's no alternative, you've gotta dig through the compilation errors yourself.

Catch configuration errors early
--------------------------------

After any change to a simulation definition, generate the input files
*without* compiling or submitting:

.. literalinclude:: snippets/troubleshooting/validate_before_submit.py
   :language: python
   :start-at: sim = picmi.Simulation

This exercises the same validation and rendering that a real run would,
so a wrong ``cfl``/time step, a bad grid distribution, a missing preset
variable, or a species/layout mismatch all fail here -- seconds instead
of a compile and a queued job.

Input-file generation
---------------------

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

Compile / CMake errors during the build step or ``pic-build`` command not found
   The toolchain (compiler, CUDA, MPI, libraries, ...) is not active or shipped binaries are not found.
   This should have been handled by your preset in ``picongpurc.toml``,
   so the most likely cause is that this was not picked up correctly.
   Verify by inspecting ``<setup_dir>/workflow/scripts/picongpu.profile``.
   It should look very similar to your preset's corresponding
   ``*_picongpu.profile.example`` (or the like)
   in https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/etc/picongpu.
   If it looks more like a stub and doesn't contain system-specific
   information, verify that your ``picongpurc.toml`` file was picked up
   correctly by putting the following at the top of your input file:

   .. code-block:: python

      from picongpu import rc_params

      for key, value in rc_params.items():
         print(f"{key:<20}: {value}")

      assert rc_params["preset"] == "<my_expected_setup>"
      exit(0)

   Check if the ``picongpurc.toml`` file exists.
   If it does, report this bug here:
   https://github.com/ComputationalRadiationPhysics/picongpu/issues/new.
   As a hotfix, try moving the ``picongpurc.toml`` file to one of the other locations described in
   :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.

During the run
--------------

Running PICMI reports workflow success but the job fails immediately and there is no sign of any artifacts, results or logs
   This is the likely failure mode when the compute node(s) do not have access to
   the filesystem where the run directory is stored.
   In the worst case, they might have a different filesystem mounted under the
   same name (e.g. a node-local or in-memory ``/tmp`` filesystem).
   The actual failure is that the submission script or some executable therein
   is not found.
   In the worst case, logs and failure messages are written to such a
   compute-node-local filesystem and are hard to recover.
   Check the paths in your ``Simulation.run()`` call.
   In particular, if no paths are given, it defaults to the temporary
   filesystem, which is particularly likely to exhibit the above behaviour.

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
