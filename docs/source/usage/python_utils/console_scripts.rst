.. _usage-python-utils-console-scripts:

Console Scripts
---------------

.. sectionauthor:: Julian J. Lenz

Installing the Python package (see :ref:`usage-python-utils`) also installs two console scripts that drive the PIConGPU runtime configuration.
Both are thin command-line frontends over the modules documented below and can alternatively be run without installing via
``uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" <script>``.

pic-deps
~~~~~~~~

``pic-deps`` is a best-effort dependency installer and checker for the preset configured in your ``.picongpurc.toml``.
It reads the preset from the standard :ref:`rc_params discovery mechanism <configuring_env_toml_file>`, renders a sourceable profile, and then either runs the preset's own ``dependencies_autoinstall.sh`` or verifies the dependency directories that script owns.

.. code-block:: bash

   pic-deps install   # run the preset's dependencies_autoinstall.sh (best-effort)
   pic-deps check     # verify each dependency directory the script owns

Neither subcommand takes further options.
``install`` always exits with the exit code of the final check; ``check`` exits with status ``1`` when at least one dependency directory is missing, so it can be used as a guard in scripts and CI.
If no preset is set (or the preset ships no ``dependencies_autoinstall.sh``) both subcommands fail with an explanatory message.

.. automodule:: picongpu.pic_deps
   :members: main, resolve_preset_script, parse_guard_roots, render_profile, check_dependencies, install_dependencies

picrc-builder
~~~~~~~~~~~~~

``picrc-builder`` is an interactive builder for ``.picongpurc.toml`` configuration files.
It asks for the preset first and then only for the variables that are still missing, so it can be used both to create a new configuration and to complete an existing one.

.. code-block:: bash

   picrc-builder                  # start from scratch, asking for a preset first
   picrc-builder path/to/config   # load and complete an existing .picongpurc.toml

The single optional positional argument is a path to an existing ``.picongpurc.toml`` to load and complete.
If the path does not exist (or points to a directory) a new configuration is created there; a directory receives a ``picongpurc.toml`` file inside it.

.. automodule:: picongpu.picrc_builder
   :members: main

See :ref:`configuring_env_picrc_builder` for the full workflow and :ref:`configuring_env_toml_file` for how the resulting file is discovered.
