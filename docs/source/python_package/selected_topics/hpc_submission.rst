.. _hpc-submission:

HPC Submission Internals
========================

This is a deep dive into what happens *after* your input files have been
generated: how the simulation is compiled, packaged and handed to the
batch system of your machine, and where everything ends up.
If you only want to run things, you do not need any of this --
:ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
covers the practical side.

The submission is performed by the `tbg <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/bin/tbg>`__
(*template batch generator*) tool from core PIConGPU,
driven by the ``TBG_*`` variables in the generated ``etc/picongpu/N.cfg``
and by a batch-script *template* for your system.
For the platform-independent parts of this tool
see :ref:`the legacy TBG documentation <usage-tbg>`.

.. _tbg-flags:

The ``tbg`` Flags
-----------------

The runner invokes ``tbg`` once to *prepare* the submission
(the ``prepare_submission`` workflow step)
and the generated ``submit.sh`` script invokes the submit command
a second time (the ``submit`` workflow step).
Two of the knobs are runtime configuration parameters
(see :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`);
the rest are flags you can pass to ``simulation.run()``.

``tbg_submit``
   The submit command to run the generated batch script:
   ``"bash"`` (local execution, the default),
   ``"sbatch"`` (Slurm), ``"qsub"`` (PBS/LSF), or a variant such as
   ``"qsub -h"``.
   Runtime configuration parameter
   (exposed to the workflow as ``submit_system``).

``tbg_tpl_file``
   The batch-script template for your system,
   e.g. ``etc/picongpu/hemera-hzdr/gpu.tpl``.
   It is the source of the ``#SBATCH``/``#PBS`` directives
   and the actual launch command of the simulation.
   Runtime configuration parameter
   (exposed to the workflow as ``template_file``).

``cfg_file``
   The configuration file to set up the batch file
   (default ``etc/picongpu/N.cfg``).

``overwrite_vars``
   A list of ``KEY=value`` strings
   that overwrite any template variable before the batch file is rendered.
   This is the supported way to tweak a submission
   without editing your system's template.

``force``
   Allow overwriting the destination directory if it already exists.

What ``tbg`` Does
-----------------

``tbg`` reads the configuration file (all the ``TBG_*`` variables,
including the grid, step count and every diagnostic flag),
reads the template, and produces the batch script.
Two mechanisms connect the two:

* lines of the form ``.Name=value`` in the template
  *define* a template variable (overriding the value from the cfg);
  these lines are stripped from the final script.
* ``!Name`` placeholders anywhere in the template
  are *substituted* with the (already computed) value of the variable.

In addition, ``tbg`` exports a small set of well-known variables
that every template can use:

``TBG_jobName``
   The name of the job (the base name of the destination directory).
``TBG_jobNameShort``
   The job name reduced to at most 15 alphanumeric characters
   (used where batch systems limit job-name length).
``TBG_cfgPath`` / ``TBG_cfgFile``
   The directory and the full path of the configuration file.
``TBG_projectPath`` / ``TBG_dstPath``
   The project (setup) path and the destination (run) directory.

The result of this step is the ``tbg/`` directory in the run directory
containing the three files shown in
:ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`:
``submit.start`` (the resolved, ready-to-run batch script),
``submit.tpl`` (the original template) and
``submit.cfg`` (a copy of the configuration used).
At this point **nothing has been submitted yet**.

The Two-Stage Submission
------------------------

The actual submission is a *separate* workflow step that runs in the
*job's own working directory* (created by ``cwltool`` in the system
temporary directory).
The generated ``submit.sh`` script:

#. copies the prepared ``tbg/`` directory next to the compiled binary
   (``input/bin``) and the run-time configuration (``input/etc``);
#. rewrites the destination path inside ``tbg/submit.start``
   (``TBG_dstPath`` and, where present, ``--chdir=...``)
   to the *actual* job working directory,
   so that the batch script writes ``simOutput/`` where it will run;
#. invokes the submit command (``tbg_submit``) on ``tbg/submit.start``.

The content of the resulting ``submission_information.txt``
depends on the submit system:
for ``bash``/``zsh`` it is the PID of the started process,
for a real batch system it is whatever that system prints
(typically the job id, e.g. ``Submitted batch job 12345``).
This file is the handle you use to manage the job afterwards.

Managing a Submitted Job
------------------------

With the job id from ``submission_information.txt`` you can query,
cancel or otherwise manage the job with your batch system's tools,
and ``link_results.sh`` connects the (temporary) job working directory
back to a location of your choice:

.. literalinclude:: ../snippets/hpc_submission/manage_submission.sh
   :language: bash

.. note::

   The job working directory lives in the system temporary directory
   and is *not* guaranteed to persist after the job finishes
   (temporary directories are cleaned on many clusters).
   Link or copy the results while the job directory still exists.

.. _tbg-templates:

Batch Templates and Profiles
----------------------------

Each supported system ships a directory under ``etc/picongpu/<system>/``
containing one or more ``*.tpl`` batch templates
(e.g. a CPU template and a GPU template)
and a ``*_picongpu.profile.example`` that documents the environment
(modules, MPI, CUDA) the submission needs.
Your runtime configuration (preset or ``picongpurc.toml``) selects
``tbg_tpl_file`` and ``tbg_submit`` accordingly;
the profile is loaded via ``PIC_PROFILE`` before the simulation runs
so that the correct compiler, MPI and library versions are active
both at build time and at run time.

When the ``tbg_submit`` command is a shell (``bash``/``zsh``),
the "batch job" is simply a locally started process
and ``submission_information.txt`` contains its PID;
when it is a real batch system (``sbatch``/``qsub``),
the submit command returns the job id and the job runs
on the compute nodes allocated by the scheduler.
