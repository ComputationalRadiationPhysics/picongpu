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

.. _configuring_env_toml_file:

The ``.picongpurc.toml`` File
-----------------------------

The runtime configuration is kept in a `TOML <https://toml.io/>`__ file
that will be read when importing the PIConGPU python package for the first time.
You can create this file by hand;
a minimal configuration just sets the preset to use on this machine (see `Presets`_ below):

.. literalinclude:: ../snippets/configuring_environment/rc_params_minimal.toml
   :language: toml

Oftentimes, it is convenient to have one ``.picongpurc.toml`` file
in a central (user-specific) location
such that any run on this specific machine can pick it up
and is automatically configured correctly.
On some occasions, project- or run-specific configurations might be necessary.
The search order described below ensures that more specific configurations take precedence,
if they are closer to the input in the directory tree.

The file is named ``.picongpurc.toml``
(with an optional ``.`` in the beginning to hide it on Unix systems)
and is searched in the following locations (first match wins)::

  1. The file pointed to by the ``PIC_RC`` environment variable (or a directory containing a ``picongpurc.toml``, if set)
  2. The first dot-prefixed ``*.picongpurc.toml`` file (e.g. ``.picongpurc.toml``) found in the current directory or any parent directory
  3. ``$XDG_CONFIG_HOME/picongpu/picongpurc.toml`` (typically ``~/.config/picongpu/picongpurc.toml``)

.. _configuring_env_rc_params:

The ``rc_params`` Object
------------------------

The PIConGPU python package's approach to runtime configuration
is inspired by `Matplotlib's rcParams <https://matplotlib.org/stable/users/explain/customizing.html>`__:
The code interacts with the runtime configuration
via a global instance of a ``dict``-like ``RCParams`` class named ``picongpu.rc_params``.
The information available in this instance at the time of querying
is the ground truth for what configuration something is executed with.

You can interact with this instance directly,
e.g. defining or reading content from it:

.. literalinclude:: ../snippets/configuring_environment/rc_params_basic.py
   :language: python
   :start-after: BEGIN-RC-BASIC
   :end-before: END-RC-BASIC

This can be useful to define, e.g., machine-specific aspects of your simulation.
Say, on a specific cluster you want to use a specific `openPMD <https://www.openpmd.org/>`__ configuration::

  OPENPMD_CONFIG = ... if "jupiter" in rc_params['preset'] else ...

This will define the ``OPENPMD_CONFIG`` in a particular way
if the string ``"jupiter"`` is found in the name of the preset
(this is a good indicator that you are running on the `JUPITER <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUPITER/JUPITER_node.html>`__ supercomputer at JSC).

The ``pic_src_path`` Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _configuring_env_pic_src_path:

The ``pic_src_path`` parameter is a special parameter.
You can override it with an explicit value
but its intention is to be automatically deduced
to point to the PIConGPU installation in use.

Presets
-------

The PIConGPU team has run on a wide variety of the largest supercomputers in the world.
For all systems we have access to, we curate a library of presets
that allow to run PIConGPU on the corresponding system.

Using Presets
^^^^^^^^^^^^^

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
You can temporarily or permanently disable this:

.. literalinclude:: ../snippets/configuring_environment/rc_params_preset_guard.py
   :language: python
   :start-after: BEGIN-RC-PRESET-GUARD
   :end-before: END-RC-PRESET-GUARD

The ``dirty_reset_policy`` can take the values ``"raise"`` (the default),
``"warn"`` or ``"ignore"``, or an arbitrary handler to finetune the behaviour.
We generally recommend to do runtime configuration via `configuring_env_toml_file`_ outside of your script.

A full list of the presets shipped with the package can be obtained via:

.. literalinclude:: ../snippets/configuring_environment/rc_params_list_presets.py
   :language: python
   :start-after: BEGIN-RC-LIST-PRESETS
   :end-before: END-RC-LIST-PRESETS

Finetuning Presets
^^^^^^^^^^^^^^^^^^

Presets can be thought of as "just setting a bunch of parameters at once".
Consequently, any of these parameters can be given another value.
Otherwise, you can inspect the ``rc_params`` instance directly to see what has been set:

.. literalinclude:: ../snippets/configuring_environment/rc_params_finetune_preset.py
   :language: python
   :start-after: BEGIN-RC-FINETUNE-PRESET
   :end-before: END-RC-FINETUNE-PRESET

The above code applies the ``rosi-hzdr`` preset,
shows all parameters that have been set on the ``rc_params`` instance
(typically by the preset)
and then adjusts the ``tbg_partition`` parameter to have a different value.
The same could have been achieved in the ``.picongpurc.toml`` file directly
because the preset is always applied first
and all other configuration modifies a given preset:

.. literalinclude:: ../snippets/configuring_environment/rc_params_finetune_preset.toml
   :language: toml

Manually Configuring Profile Content
------------------------------------

The main functionality provided by the runtime configuration is
providing a machine-specific environment to run the code in.
Upon execution, PIConGPU's Python frontend generates
self-contained scripts to run all the different steps (compilation, submission, ...)
as well as a general profile that can be sourced to drop into the PIConGPU environment.

Any of the above can be overridden using an ``rc_params`` entry, e.g.:

.. literalinclude:: ../snippets/configuring_environment/rc_params_shebang.toml
   :language: toml

and are given reasonable defaults otherwise.

A manual configuration of the ``profile_content`` can be useful for running on a system
for which we do not provide a preset yet.
The ``profile_content`` is determined by the following cascade of prioritized defaults:

  1. A literal ``profile_content`` value in rc_params
  2. The content of a file referenced by ``profile_path``
  3. A ``profile_template_content`` string rendered as a `mustache <https://mustache.github.io/>`__ template using ``rendering_context``
  4. The content of a file referenced by ``profile_template_path``, rendered as a mustache template
  5. A minimal profile that only adds the PIConGPU tools to ``$PATH`` (insufficient for running)

The following list gives a redundant configuration with strictly decreasing precedence:

.. literalinclude:: ../snippets/configuring_environment/rc_params_profile_precedence.toml
   :language: toml

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
