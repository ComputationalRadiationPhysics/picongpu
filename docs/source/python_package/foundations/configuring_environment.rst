Configuring Your Environment
============================

PIConGPU is run in a heterogeneous HPC landscape on a daily basis.
This is enabled by -- among other things --
a clear separation between environment definition and user input.
This section is concerned with the environment configuration
which is typically tailored to a specific machine and user,
sometimes also to a particular project.
This is in contrast to the :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>` chapter,
which is used to specify the simulations and physical intent
independent of the machine, user, project, ... running this.
We use the name "runtime configuration" for all aspects orthogonal to simulation definition.
This includes aspects that in C/C++ jargon are considered "compiletime".

At the time of writing, the runtime configuration is used for the following aspects:

  * On a specific machine making the correct compilers, libraries, etc. available.
  * For a specific user configuring the correct metadata to facilitate FAIR workflows.

.. _configuring_env_picrc_builder:

Recommended: The ``picrc-builder`` tool
=======================================

In order to streamline the onboarding onto a new system,
we provide the ``picrc-builder`` tool.
It guides you through the process of writing a runtime configuration file ``.picongpurc.toml``.

It allows you to pick your choice from the list of available presets
and guides you through filling in required parameters as well as finetuning the additional ones.
You can run it via::

  picrc-builder

from any environment in which the PIConGPU python package is installed.
You can run it without installation via, e.g., the `uv <https://docs.astral.sh/uv/>`__ tool::

  uv run --with="picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" picrc-builder

Make sure to save the generated file in one of the locations listed in `configuring_env_toml_search`_,
when you actually run your simulation.


``rc_params`` and the ``.picongpurc.toml`` file
========================================================

The PIConGPU python package's approach to runtime configuration
is inspired by `Matplotlib's rcParams <https://matplotlib.org/stable/users/explain/customizing.html>`__:
The code interacts with the runtime configuration
via a global instance of a ``dict``-like ``RCParams`` class named ``picongpu.rc_params``.
The information available in this instance at the time of querying
is the ground truth for what configuration something is executed with.

The ``rc_params`` object
------------------------

You can interact with this instance directly, e.g., defining or reading content from it::

  from picongpu import rc_params

  rc_params['my_cool_config'] = 42

  if rc_params['my_cool_config'] == 42:
    print("It worked!")

This can be useful to define, e.g., machine-specific aspects of your simulation.
Say, on a specific cluster you want to use a specific `openPMD <https://www.openpmd.org/>`__ configuration::

  OPENPMD_CONFIG = ... if "jupiter" in rc_params['preset'] else ...

This will define the ``OPENPMD_CONFIG`` in a particular way
if the string ``"jupiter"`` is found in the name of the preset
(this is a good indicator that you are running on the `JUPITER <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUPITER/JUPITER_node.html>`__ supercomputer at JSC).

.. _configuring_env_toml_search:

The ``.picongpurc.toml`` file
-----------------------------

More generally, however, we expect you to keep
your runtime configuration separate from your input files.
In order to do so, you can define it in a `TOML <https://toml.io/>`__ file
that will be read when importing the PIConGPU python package for the first time.
This file is named ``.picongpurc.toml``
(with an optional ``.`` in the beginning to hide it on Unix systems)
and can be located in one of the following locations (searched in order of decreasing precedence)::

  search order (first match wins):
  1. The file pointed to by the ``PIC_RC`` environment variable (if set)
  2. The first ``*.picongpurc.toml`` or ``.picongpurc.toml`` found in the current directory or any parent directory
  3. ``$XDG_CONFIG_HOME/picongpu/picongpurc.toml`` (typically ``~/.config/picongpu/picongpurc.toml``)

Oftentimes, it is convenient to have one ``.picongpurc.toml`` file
in a central (user-specific) location
such that any run on this specific machine can pick it up
and is automatically configured correctly.

On some occasions, project- or run-specific configurations might be necessary.
The above search order ensures that more specific configurations take precedence,
if they are closer to the input in the directory tree.

The ``pic_src_path`` parameter
------------------------------

.. _configuring_env_pic_src_path:

The ``pic_src_path`` parameter is a special parameter.
You can override it with an explicit value
but its intention is to be automatically deduced
to point to the PIConGPU installation in use.
    
    
Presets
=======

The PIConGPU team has run on a wide variety of the largest supercomputers in the world.
For all systems we have access to, we curate a library of presets
that allow to run PIConGPU on the corresponding system.

Using presets
-------------

Presets are special keys in ``rc_params``.
Setting them will reset the ``rc_params`` instance
and load default values for various configuration parameters.

Presets require some parameters to be set explicitly.
Attempting to use a preset without those being set
raises an exception stating the offending parameter name
and listing all required parameters.
You can then go ahead and configure those parameters explicitly.

Due to their destructive nature setting a preset from within a script
is guarded against by a policy.
You can temporarily or permamently disable this via::

  # might raise an exception when any non-default configuration was applied already:
  rc_params['preset'] = 'bash'

  # temporarily disable that exception:
  with rc_params.set_temporarily('dirty_reset_policy', 'ignore'):
    rc_params['preset'] = 'bash'

  # permamently disable that exception:
  rc_params['dirty_reset_policy'] = 'warn'

The ``dirty_reset_policy`` can take an arbitrary handler to finetune the behaviour.
We generally recommend to do runtime configuration via `configuring_env_toml_search`_ outside of your script.

A full list of presets can be obtained via::

  picongpu.get_available_presets()


Finetuning presets
------------------

Presets can be thought of as "just setting a bunch of parameters at once".
Consequently, any of these parameters can be given another value.
The `configuring_env_picrc_builder`_ allows you to change these values.
Otherwise, you can inspect the ``rc_params`` instance directly to see what has been set::

  for key, value in rc_params.items():
    print(f'{key}: {value}')

  # changing a default
  rc_params['tbg_partition'] = 'a100'

The above code shows all parameters that have been set on the ``rc_params`` variable
(typically by the preset)
and then adjusts the `tbg_partition` parameter to have a different value.
The same could have been achieved in the ``.picongpurc.toml`` file directly
because the preset is always applied first
and all other configuration modifies a given preset::

  preset = "rosi-hzdr"
  tbg_partition = "a100"
  # ...

Manually configuring profile content
====================================

The main functionality provided by the runtime configuration is
providing a machine-specific environment to run the code in.
Upon execution, PIConGPU's Python frontend generates
self-contained scripts to run all the different steps (compilation, submission, ...)
as well as a general profile that can be sourced to drop into the PIConGPU environment.

Any of the above can be overridden using an ``rc_params`` entry, e.g.,::

  shebang = "#!/usr/bin/env zsh"

and are given reasonable defaults otherwise.

A manual configuration of the ``profile_content`` can be useful for running on a system
for which we do not provide a preset yet.
The ``profile_content`` is determined by the following cascade of prioritized defaults:

  1. A literal ``profile_content`` value in rc_params
  2. The content of a file referenced by ``profile_path``
  3. A ``profile_template_content`` string rendered as a `mustache <https://mustache.github.io/>`__ template using ``rendering_context``
  4. The content of a file referenced by ``profile_template_path``, rendered as a mustache template
  5. A minimal profile that only adds the PIConGPU tools to ``$PATH`` (insufficient for running)

The following list gives a redundant configuration with strictly decreasing precedence::

  # setting a custom parameter up front:
  my_rc_params_value = "Rendering template content directly"

  profile_content = "echo 'Using profile_content directly'"
  profile_path = "/path/to/my/profile"
  profile_template_content = "echo {my_rc_params_value}"
  profile_template_path = "/path/to/my/profile-template"

The above configuration has the following effect:

  * As given above, the ``profile_content`` would be ``echo 'Using profile_content directly'``.
    ``profile_content`` has the highest precedence
    and all other lines are ignored.
  * Removing the first line
    would make it read ``/path/to/my/profile`` and use that as ``profile_content``.
  * Removing the second line as well
    would make it render the given string in ``profile_template_content`` as a mustache template.
    Considering the custom parameter at the top,
    the result would be ``echo Rendering template content directly``.
  * Removing all but the last line would read the content of ``/path/to/my/profile-template``
    and render that content as a mustache template.
  * Without any of the above, the ``profile_content`` would only modify the ``$PATH``
    to make PIConGPU tools available.
    In any but some edge cases, this will be insufficient to run your code
    and will result in interesting errors.

We have seen that we can define and use arbitrary parameters in our templates.
We can define them in our ``.picongpurc.toml`` file (or via ``rc_params[...] = ...``).
If an undefined variable is encountered,
the ``missing_variable_policy`` is called to determine how to proceed.
By default it raises an exception.
The special variable ``pic_src_path`` can be used to refer to
the installation path of PIConGPU itself (see `configuring_env_pic_src_path`_ above).
