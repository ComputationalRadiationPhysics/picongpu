Verifying Your Simulation
=========================

How to gain confidence that your setup is correct, in stages of increasing
cost.
Work from the cheapest stage up:
each one catches a different class of problem,
and the early stages are fast enough to run after *every* change.

1. Is the package installed?
----------------------------

On any machine where you intend to use the package,
check that it imports and exposes the PICMI frontend:

.. literalinclude:: ../snippets/verification/verify_install.sh
   :language: bash

2. Is my simulation definition valid?
-------------------------------------

Generate the input files *without* compiling or submitting:

.. literalinclude:: ../snippets/troubleshooting/validate_before_submit.py
   :language: python
   :start-after: setup_dir = Path

This runs the same validation and template rendering that a real run would,
so a wrong time step, a bad grid distribution, a species/layout mismatch,
or a missing preset variable all fail here -- in seconds,
before you spend a compile or a queued job.
See :ref:`Troubleshooting <python_package/selected_topics/troubleshooting:Troubleshooting>`
for what each of these errors means.

You can then inspect the generated files
(``include/picongpu/param/*.param``, ``etc/picongpu/N.cfg``)
to confirm the parameters are what you intended.

3. Does the frontend work on this machine?
------------------------------------------

A source checkout of PIConGPU ships a fast test suite that exercises the
Python frontend without a GPU or a compiler
(the test tree is not part of the pip/uv installation,
so this stage needs a checkout, not an install):

.. literalinclude:: ../snippets/verification/quick_suite.sh
   :language: bash

These are the "quick" tests: they run in seconds and are the ones CI runs
on every commit.
A green run here means the frontend is internally consistent.
(Deeper suites -- ``compiling/`` and ``end_to_end/`` -- build and run real
simulations and are long-running; they are for the project's CI,
not for a quick user check.)

4. Does it build?
-----------------

Running the simulation (``simulation.run()``) first compiles a binary
tailored to your input files
(see :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`).
A successful build means the generated parameters are internally consistent
and your toolchain (compiler, CUDA, MPI) is active.
A build failure is almost always an environment problem
(a missing module, a wrong ``PICSRC``/``PIC_BACKEND``),
not a mistake in your physics definition.

5. Does it produce what I expect?
---------------------------------

After a run, confirm the diagnostics you asked for actually exist
in the output (see the :ref:`diagnostics pages <python_package/selected_topics/index:Selected Topics>`
for where each one writes):
the openPMD files in ``simOutput/openPMD/``,
the phase-space files in ``simOutput/phaseSpace/``,
the energy-histogram / macro-particle-count files in ``simOutput/``,
and so on.
A run that finishes but writes none of the diagnostics you configured
usually means a ``period`` that never fires
(e.g. a period beyond the number of steps you actually ran).
