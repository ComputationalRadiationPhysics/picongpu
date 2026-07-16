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
In this case, you can use::

  uv run my_input.py
  pip-run my_input.py
  hatch run my_input.py

  # With the corresponding shebang, e.g.,
  #   #!/usr/bin/env -S uv run
  # and `chmod +x my_input.py` you can even do:
  ./my_input.py

Any of these will download the specified version of PIConGPU (and other dependencies),
generate the necessary input files
and immediately execute the necessary steps (i.e. the workflow)
to build and run PIConGPU on the configured system
(if you've used ``simulation.run()`` and not ``simulation.write_input_file()``).
CAUTION: ``uv run`` (and potentially others) might try to set a file lock.
This can cause problems on parallel file system.
Use one of the other methods below if you run into trouble.

For most HPC systems, this means that we'll submit (at least) the main simulation job
to a set of dedicated compute nodes.
In its current form, the interface returns after submission
and does not further monitor the progress of the submitted job.
In the specified run directory you will find two pieces of information
about your submission:

#. ``submission_information.txt`` contains sufficient information to uniquely identify the submitted batch job.
   You can use that information to monitor progress, etc.
#. ``link_results.sh`` is a shell script that can be used to link the results of your simulation to a specified location::

    # general linking to a user-defined location
    $RUN_DIR/link_results.sh /path/to/my/results

    # default linking, restoring the behaviour as found in the legacy workflow
    cd $RUN_DIR
    ./link_results.sh

  You can also just read the script to find out where your data ended up.

.. _running_simulation_from_installation:

From Installation
^^^^^^^^^^^^^^^^^

Under some circumstances,
it might be beneficial to install PIConGPU manually
(e.g., in order to harmonise the version used in a project).
You can install PIConGPU via ``pip`` via::

  pip install "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"

We recommend to replace the ``@dev`` with a specific ``@<commit hash>`` to fix the version.
We recommend to install into a `virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`__ (e.g. via `venv <https://docs.python.org/3/library/venv.html>`__, `uv <https://docs.astral.sh/uv/>`__, `mamba <https://mamba.readthedocs.io/>`__, ...)

This has downloaded the full source code of PIConGPU under the hood
and has made the Python library and tooling available.
You can simply run the script as Python script (from your environment)::

  python my_input.py

and proceed as above.

From Source
^^^^^^^^^^^

From the full source code (e.g. a clone of the repository)
you can install the Python package via::

  pip install -e lib/python

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

In your PICMI input script you can use::

  simulation.write_input_file()

instead of ``simulation.run()``
to write a simulation setup the specified location
without executing it.

If you are familiar with the legacy ``pic-create``/``pic-build``/``tbg`` interface of core PIConGPU (:ref:`TBG documentation <usage-tbg>`),
you can use the generated setup in the same manner that you would have used a ``pic-create`` setup.
Furthermore, you can find a tailored :ref:`profile <install-profile>` in ``workflow/scripts/picongpu.profile``.
In effect, you can run::

  cd $SETUP_DIR
  source workflow/scripts/picongpu.profile
  pic-build
  tbg $TBG_ARGS $RUN_DIR

to achieve roughly the same result that a call to ``simulation.run()`` would have had.
You will still benefit in parts from the additional features like better metadata, etc.

Manual and Partial Workflow Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manually running the full workflow
""""""""""""""""""""""""""""""""""

Starting from a generated setup (see `running_simulation_legacy_workflow`_),
we can find a full workflow definition in `Common Workflow Language (CWL) <https://www.commonwl.org/>`__ in ``workflow/``.
The exact equivalent of using ``simulation.run()`` directly
can be achieved on a generated setup the following invocation of the `cwltool <https://github.com/common-workflow-language/cwltool>`__::

  CWL_ARGS="--leave-tmpdir --preserve-entire-environment --cachedir=.cwl_cache"
  cd $RUN_DIR
  cwltool $CWL_ARGS $SETUP_DIR/workflow/workflow.cwl $SETUP_DIR/workflow/input.yaml
  ./link_results.sh

In here, the ``workflow/workflow.cwl`` contains the full definition of
the workflow of building and submitting your simulation.
``workflow/input.yaml`` and ``CWL_ARGS``
contain the input parameters resp. ``cwltool`` runtime context
as the default orchestration via ``simulation.run()`` would have used them.
You can use them as customization points to meet your specific needs.

Running individual steps of the workflow
""""""""""""""""""""""""""""""""""""""""

The ``workflow/workflow.cwl`` refers to individual steps as defined in ``workflow/steps/``.
These can be executed individually in the following manner (exemplified by the ``build.cwl`` step)::

  cd $RUN_DIR

  # either provide the necessary input definitions on the commandline
  # (e.g. ``build.cwl`` requires at least ``include_directory`` and ``script``):
  cwltool $CWL_ARGS $SETUP_DIR/workflow/steps/build.cwl --include_directory $SETUP_DIR/include --script $SETUP_DIR/workflow/scripts/build.sh

  # or write a custom ``my_input.yaml`` file with content like:
  # include_directory: <SETUP_DIR>
  # script: <SETUP_DIR>/workflow/scripts/build.sh
  cwltool $CWL_ARGS $SETUP_DIR/workflow/steps/build.cwl my_input.yaml

Running the individual scripts manually
"""""""""""""""""""""""""""""""""""""""

The individual workflow steps refer to generated bash scripts to do their job.
Those can be invoked directly as well.
The ``InitialWorkDirRequirement`` section of a workflow step contains information about
how to re-create a clean working directory as ``cwltool`` would do it upon execution.
For example, the ``build.cwl`` specifies that it needs access to the ``include/`` directory and the ``workflow/scripts/build.sh`` script.
Consequently, we can perform the equivalent of the above partial workflow execution via::

  mkdir $RUN_DIR/build_step
  cd $RUN_DIR/build_step
  ln -s $SETUP_DIR/include
  ln -s $SETUP_DIR/workflow/scripts/build.sh
  ./build.sh

Integration into overarching workflows
--------------------------------------
