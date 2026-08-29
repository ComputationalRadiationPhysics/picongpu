Running Your Simulation
=======================

A PICMI input file is a simple Python script.
As such, any way of executing a Python script works for running your simulation.
In this section, we have compiled the most convenient and useful ways first
and look at some advanced usecases afterwards.

In order to run a PIConGPU simulation, you need

#. a PICMI input file (see :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>`) in which
   ``simulation.run()`` (or ``simulation.write_input_file()`` if that's what you want to do)
   gets called eventually and
#. a valid runtime configuration (see :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`).

The user should be aware
that nothing prevents us from calling ``simulation.run()`` or ``simulation.write_input_file()``
multiple times in the same script (see :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>` for inspirations on how to use this).
For simplicity, the following guide assumes that a single simulation setup/run is handled
but the concepts apply equally to multi-simulation scripts.

Full Execution
--------------

Recommended: From Script Inline Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As discussed in the :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>` section,
we recommend to use `PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__ in your input files
to document and fix the version of PIConGPU you are running.
In this case, you can use one of the following:

.. tab-set::

    .. tab-item:: uv

        `uv <https://docs.astral.sh/uv/>`__ is a fast Python package installer
        and runner, written in Rust.
        Install it with

        .. literalinclude:: ../snippets/running_simulation/uv_install.sh
           :language: bash
           :start-after: BEGIN-UV-INSTALL
           :end-before: END-UV-INSTALL

        or follow the installation instructions on `GitHub <https://github.com/astral-sh/uv#installation>`__.

        Once installed, run

        .. literalinclude:: ../snippets/running_simulation/uv_run.sh
           :language: bash
           :start-after: BEGIN-UV-RUN
           :end-before: END-UV-RUN

    .. tab-item:: pip-run

        `pip-run <https://pip-tools.readthedocs.io/>`__ is part of the ``pip-tools``
        project, a small set of plugins around ``pip``
        for running scripts with transient, isolated dependencies.
        Install it with

        .. literalinclude:: ../snippets/running_simulation/pip_run_install.sh
           :language: bash
           :start-after: BEGIN-PIP-RUN-INSTALL
           :end-before: END-PIP-RUN-INSTALL

        Then run

        .. literalinclude:: ../snippets/running_simulation/pip_run.sh
           :language: bash
           :start-after: BEGIN-PIP-RUN
           :end-before: END-PIP-RUN

    .. tab-item:: hatch

        `hatch <https://hatch.pypa.io/>`__ is a Python project manager
        that also supports running scripts from a single file using the
        `hatch-run <https://hatch.pypa.io/latest/config/cli/#run>`__ plugin.
        Install it with

        .. literalinclude:: ../snippets/running_simulation/hatch_install.sh
           :language: bash
           :start-after: BEGIN-HATCH-INSTALL
           :end-before: END-HATCH-INSTALL

        Then run

        .. literalinclude:: ../snippets/running_simulation/hatch_run.sh
           :language: bash
           :start-after: BEGIN-HATCH-RUN
           :end-before: END-HATCH-RUN

    .. tab-item:: executable shebang

        With a suitable shebang (e.g., ``#!/usr/bin/env -S uv run``)
        and ``chmod +x my_input.py`` you can also run it directly:

        .. literalinclude:: ../snippets/running_simulation/executable_shebang.sh
           :language: bash
           :start-after: BEGIN-EXECUTABLE-SHEBANG
           :end-before: END-EXECUTABLE-SHEBANG

Any of these will download the specified version of PIConGPU (and other dependencies),
generate the necessary input files
and immediately execute the necessary steps (i.e. the workflow)
to build and submit PIConGPU on the configured system
(if you've used ``simulation.run()`` and not ``simulation.write_input_file()``).

.. warning::

   ``uv run`` (and potentially others) might try to set a file lock.
   This can cause problems on parallel file systems.
   Use one of the other methods below if you run into trouble.

For most HPC systems, this means that we'll submit (at least) the main simulation job
to a set of dedicated compute nodes.
In its current form, the interface returns after submission
and does not further monitor the progress of the submitted job.
In the specified run directory you will find two pieces of information
about your submission:

#. ``submission_information.txt`` contains sufficient information to uniquely identify the submitted batch job.
   You can use that information to monitor progress, etc.
#. ``link_results.sh`` is a shell script that can be used to link the results of your simulation to a specified location:

    .. literalinclude:: ../snippets/running_simulation/link_results.sh
       :language: bash
       :start-after: BEGIN-LINK-RESULTS
       :end-before: END-LINK-RESULTS

    You can also just read the script to find out where your data ended up.

The Workflow
------------

``simulation.run()`` orchestrates the build and submission as a workflow
in the `Common Workflow Language (CWL) <https://www.commonwl.org/>`__
(version 1.2),
which is written into the setup directory as ``workflow/workflow.cwl``.
The workflow consists of four steps:

1. **build**
   Runs the generated script ``workflow/scripts/build.sh``,
   which sets up the environment from your runtime configuration
   and compiles the tailored PIConGPU binary via ``pic-build``.
   The compiled binaries end up in ``bin/``.
2. **prepare submission**
   Runs the generated script ``workflow/scripts/prepare_submission.sh``,
   which invokes ``tbg`` with the configured batch parameters
   (see `Advanced Workflows`_ for the available knobs)
   and creates the batch script
   (``tbg/submit.start``, generated from the configured template and
   configuration files ``tbg/submit.tpl`` and ``tbg/submit.cfg``).
   At this point, nothing is submitted yet.
3. **submit**
   Runs the generated script ``workflow/scripts/submit.sh``,
   which pins the output location of the batch script to its working directory
   and submits it with the configured submission command
   (``tbg_submit`` of your preset, e.g. ``sbatch`` for Slurm
   or ``bash`` for local execution).
   This produces the ``submission_information.txt`` and ``link_results.sh``
   mentioned above.
4. **organize output**
   Runs the generated script ``workflow/scripts/organize_output.sh``,
   which assembles the final run directory (see below).

All inputs of the workflow steps
(the build parameters and the ``tbg`` parameters)
are written into ``workflow/input.yaml``
and can be customized by passing them as keyword arguments
to ``simulation.run()`` / ``simulation.write_input_file()``
(e.g. ``simulation.run(setup_dir=..., run_dir=..., jobs=8, force=True)``).

The Layout of the Run Directory
-------------------------------

After ``simulation.run()`` returned,
the run directory looks like this::

  my_run/
  ├── input/                      # the generated setup directory (a copy)
  │   ├── bin/                    #   the compiled PIConGPU binaries
  │   ├── etc/                    #   runtime parameters (incl. the preset's templates)
  │   ├── include/                #   compile-time parameters
  │   ├── metadata/               #   machine-readable metadata (see below)
  │   └── workflow/               #   the workflow definition and the scripts
  ├── tbg/                        # the generated batch script and its inputs
  │   ├── submit.start            #   the resolved batch script
  │   ├── submit.tpl              #   the template it was generated from
  │   └── submit.cfg              #   the configuration file it was generated from
  ├── submission_information.txt  # the id of the submitted job
  ├── link_results.sh             # links the simulation output to a location of your choice
  └── .cwl_cache/                 # the step cache of the workflow engine (see below)

The simulation itself is executed by the batch script in its working directory,
which the workflow engine (``cwltool``) creates in the system's temporary directory
(default: ``/tmp/``) and keeps after the workflow has finished
(the destination path the submission was pinned to).
The output is written to ``simOutput/`` in that directory.
``link_results.sh`` encodes the absolute path of that location
and creates a symbolic link to its ``simOutput/`` at a path you choose
(you can also just read the script to find out where your data ended up).

Metadata and Provenance
-----------------------

Every generated setup directory carries machine-readable descriptions
of itself, so that it stays self-contained and reusable:

* ``metadata/pypicongpu_runner.json``:
  the state of the runner,
  including the complete PyPIConGPU representation of the simulation.
* ``metadata/pypicongpu_rendering_context.json``:
  the full rendering context,
  i.e. all the values that went into the template rendering.
* ``metadata/rc_params.json``:
  the runtime configuration that was used.
* ``ro-crate-metadata.json``:
  an `RO-Crate <https://www.w3.org/TR/2021/REC-vc-r-crate-20211109/>`__ (version 1.2)
  describing the setup directory as a dataset:
  the ``workflow/workflow.cwl`` as its main entity,
  the PIConGPU version as the software used,
  and descriptions of the ``etc/``, ``include/``, ``metadata/`` and ``workflow/`` sub-datasets.
  This is what makes the setup understandable to (and processable by)
  tools that speak RO-Crate,
  independently of the PIConGPU Python package.

Resubmitting and Restarting
---------------------------

The workflow is executed with `cwltool <https://github.com/common-workflow-language/cwltool>`__,
which caches the outputs of the steps in ``.cwl_cache/`` inside the run directory.
If you re-run the same workflow
(e.g. after adjusting an input parameter),
steps whose inputs have not changed are reused from the cache
instead of being re-executed
(only the steps that depend on the changed input,
and the steps after them, run again).
If you want to force a full re-execution,
delete ``.cwl_cache/`` first.

Regarding restarting a simulation that has already been started:
the Python interface currently runs simulations from scratch only.
In particular, ``simulation.step(n)``
is not a way to run a simulation in increments:
it requires ``n`` to be the full length of the simulation (``max_steps``)
and executes the complete workflow for it.
(There is a pending effort in the code to support incremental runs/restarts
through the runner;
until then, a checkpoint-based restart
in the sense of the C++ code
has to be arranged outside of the Python interface.)

.. _running_simulation_from_installation:

From Installation
^^^^^^^^^^^^^^^^^

Under some circumstances,
it might be beneficial to install PIConGPU manually
(e.g., in order to harmonise the version used in a project).
You can install PIConGPU via ``pip`` via:

.. literalinclude:: ../snippets/running_simulation/pip_install_from_git.sh
   :language: bash
   :start-after: BEGIN-PIP-INSTALL-FROM-GIT
   :end-before: END-PIP-INSTALL-FROM-GIT

We recommend to replace the ``@dev`` with a specific ``@<commit hash>`` to fix the version.
We recommend to install into a `virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`__ (e.g. via `venv <https://docs.python.org/3/library/venv.html>`__, `uv <https://docs.astral.sh/uv/>`__, `mamba <https://mamba.readthedocs.io/>`__, ...)

This has downloaded the full source code of PIConGPU under the hood
and has made the Python library and tooling available.
You can simply run the script as Python script (from your environment):

.. literalinclude:: ../snippets/running_simulation/python_input.sh
   :language: bash
   :start-after: BEGIN-PYTHON-INPUT
   :end-before: END-PYTHON-INPUT

and proceed as above.

From Source
^^^^^^^^^^^

From the full source code (e.g. a clone of the repository)
you can install the Python package via:

.. literalinclude:: ../snippets/running_simulation/pip_install_from_source.sh
   :language: bash
   :start-after: BEGIN-PIP-INSTALL-FROM-SOURCE
   :end-before: END-PIP-INSTALL-FROM-SOURCE

Make sure to use ``-e`` in order for the installation
to take into account changes in your repository.
Afterwards, you can proceed as in `running_simulation_from_installation`_.
This is intended for development purposes.
Developers should also look into the ``pyproject.toml`` file
at `lib/python/pyproject.toml <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/lib/python/pyproject.toml>`__
to find out about optional dependencies (like test or development dependencies).

Advanced Workflows
------------------

PIConGPU's Python package can take full control of orchestrating the various steps for running your simulation.
But under specific circumstances, more fine-grained control for the user is required.
For such cases, the following workflows are supported.
The following assumes that the variables ``$SETUP_DIR`` and ``$RUN_DIR`` are set to the same values
that they would have in the equivalent ``simulation.run()`` invocation.

.. _running_simulation_legacy_workflow:

Input for the Legacy Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your PICMI input script you can use ``simulation.write_input_file()``
instead of ``simulation.run()``
to write a simulation setup the specified location
without executing it.

If you are familiar with the legacy ``pic-create``/``pic-build``/``tbg`` interface of core PIConGPU (:ref:`TBG documentation <usage-tbg>`),
you can use the generated setup in the same manner that you would have used a ``pic-create`` setup.
Furthermore, you can find a tailored :ref:`profile <install-profile>` in ``workflow/scripts/picongpu.profile``.
In effect, you can run:

.. literalinclude:: ../snippets/running_simulation/legacy_workflow.sh
   :language: bash
   :start-after: BEGIN-LEGACY-WORKFLOW
   :end-before: END-LEGACY-WORKFLOW

to achieve roughly the same result that a call to ``simulation.run()`` would have had.
You will still benefit in parts from the additional features like better metadata, etc.

Manual and Partial Workflow Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manually running the full workflow
""""""""""""""""""""""""""""""""""

Starting from a generated setup (see `running_simulation_legacy_workflow`_),
we can find a full workflow definition in `Common Workflow Language (CWL) <https://www.commonwl.org/>`__ in ``workflow/``.
The exact equivalent of using ``simulation.run()`` directly
can be achieved on a generated setup the following invocation of the `cwltool <https://github.com/common-workflow-language/cwltool>`__:

.. literalinclude:: ../snippets/running_simulation/cwltool_workflow.sh
   :language: bash
   :start-after: BEGIN-CWLTOOL-WORKFLOW
   :end-before: END-CWLTOOL-WORKFLOW

In here, the ``workflow/workflow.cwl`` contains the full definition of
the workflow of building and submitting your simulation.
``workflow/input.yaml`` and ``CWL_ARGS``
contain the input parameters resp. ``cwltool`` runtime context
as the default orchestration via ``simulation.run()`` would have used them.
You can use them as customization points to meet your specific needs.

Running individual steps of the workflow
""""""""""""""""""""""""""""""""""""""""

The ``workflow/workflow.cwl`` refers to individual steps as defined in ``workflow/steps/``.
These can be executed individually in the following manner (exemplified by the ``build.cwl`` step):

.. literalinclude:: ../snippets/running_simulation/cwltool_step.sh
   :language: bash
   :start-after: BEGIN-CWLTOOL-STEP
   :end-before: END-CWLTOOL-STEP

Running the individual scripts manually
"""""""""""""""""""""""""""""""""""""""

The individual workflow steps refer to generated bash scripts to do their job.
Those can be invoked directly as well.
The ``InitialWorkDirRequirement`` section of a workflow step contains information about
how to re-create a clean working directory as ``cwltool`` would do it upon execution.
For example, the ``build.cwl`` specifies that it needs access to the ``include/`` directory and the ``workflow/scripts/build.sh`` script.
Consequently, we can perform the equivalent of the above partial workflow execution via:

.. literalinclude:: ../snippets/running_simulation/manual_step.sh
   :language: bash
   :start-after: BEGIN-MANUAL-STEP
   :end-before: END-MANUAL-STEP

Integration into overarching workflows
--------------------------------------

If you want to embed PIConGPU runs into your own workflows
(a parameter sweep, a follow-up simulation that depends on the output of a previous one,
a cluster-specific pipeline, ...),
the generated setup directory is designed to be the integration point:

* It is **self-contained**:
  the runtime configuration is baked into the scripts,
  the workflow is a standard CWL 1.2 definition
  (drivable by any CWL engine,
  not only the ``cwltool`` invocation that ``simulation.run()`` uses),
  and the scripts can be invoked directly
  (see `Running the individual scripts manually`_).
* It is **self-describing**:
  the ``metadata/`` directory and the RO-Crate
  (see `Metadata and Provenance`_)
  capture the simulation and the environment completely,
  so that other tools (and your future self) can inspect and reuse it.
* It is **parameterized**:
  ``workflow/input.yaml`` contains all inputs of the workflow steps
  as plain data,
  which you can modify and feed to your own workflow engine.

A typical integration strategy is to call ``simulation.write_input_file()``
once per simulation in your orchestrating script
and to drive the individual steps (or the full workflow) from there,
using the run directory as the exchange format between your steps.
