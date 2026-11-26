.. _conquering_new_cluster-integration:

The General Process
===================

.. sectionauthor:: Julian Lenz

PIConGPU provides tools on various levels to streamline the process of compiling and running it.
Regardless of whether you're using our modern :ref:`PICMI Interface <python_package/foundations/index:Foundations>` or the :ref:`Low-level C++ interface <usage-index>`
a new system needs the following pieces of information:

#. Environment information in the form of a
   :ref:`picongpu.profile <install-profile>`
#. Submission information in the form of a
   :ref:`batch template <conquering_new_cluster-templates>` (roughly a template for a batch script)
#. An installation of :ref:`dependencies <install-dependencies>`
   that are not provided by the system itself

Authoring a custom profile
--------------------------

The following process is suggested for arriving at a custom profile.

0. Read :ref:`picongpu.profile <install-profile>` in order to gain a basic
   understanding of the following edits.
1. Go through
   `etc/picongpu <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/etc/picongpu>`__
   and look for a system similar to the one you're working with
   (e.g. in terms of batch system, hardware, ...).
2. Copy ``*_picongpu.profile.example`` as a starting point.
   We'll call this ``custom.profile`` in the following.
3. Figure out which of the :ref:`dependencies <install-dependencies>` are
   provided by the system, typically in form of a module system.
   Commands like ``module avail`` and your system's documentation will
   probably be helpful.
4. Edit your copy of the profile to load all your system-provided dependencies
   and take a note of the unfulfilled dependencies.
5. Check the remaining content and adjust system-specific details
   (e.g. which hardware to compile for, ...).

If you're planning to use the PICMI frontend also consider:

6. Optional (if some dependencies are not provided by the system):
   Make sure it exports ``PIC_LIBS={{{pic_libs}}}`` and the corresponding
   dependency root environment variables (e.g. ``BOOST_ROOT=$PIC_LIBS/...``, ...)
   making those dependencies available.
7. Replace the line ``export PICSRC=...`` with
   ``export PICSRC={{{pic_src_path}}}``


.. _conquering_new_cluster-templates:

Authoring a custom batch template
---------------------------------

PIConGPU separates the portable run configuration (``.cfg``) from the
machine-dependent batch script (``.tpl``).
A template is the job script for one queue:
it carries the scheduler directives (``#SBATCH``, ``#PBS``, ...),
computes the resources from the requested grid and task count,
and launches the executable.
See :ref:`the low-level TBG documentation <usage-tbg>` for the mechanism and
:ref:`HPC Submission Internals <hpc-submission>` for how the frontend drives it.

We ship templates for the curated systems under ``etc/picongpu/<system>/``
together with generic launchers
``etc/picongpu/bash/mpiexec.tpl`` and ``etc/picongpu/bash/mpirun.tpl``
for local and interactive runs.
As with the custom profile, choose a system similar to yours as
a starting point for system-specific adjustments.


.. _conquering_new_cluster-manual-installation:

Installing missing dependencies
-------------------------------

Familiarize yourself with our set of :ref:`dependencies <install-dependencies>`
and make a list of what's missing.
The scripts for local installation of dependencies from ``etc/picongpu/bash/``
(``dependencies_autoinstall.sh``, ``dependencies_autoinstall_spack.sh`` and
``dependencies_autoinstall_conda.sh``)
provide a good starting point/inventory of installation routines.
Sometimes, individual compiler flags, etc. might need changing.
System-specific examples can be found in other parts of ``etc/picongpu``.
You can even try to run one of them directly on your new system
(potentially via :ref:`pic-deps <conquering_new_cluster-configuring-frontend>`).

CAUTION: Make sure that which ever process you end up with
builds the missing dependencies against the system-provided ones,
particularly MPI.
On most systems only the system-provided MPI works
and achieves full performance.
If your system explicitly provides ``spack`` or ``easybuild`` for individual users,
that's probably your best guess.
Otherwise they are hard to configure to take external dependencies into account.


This page is basically the practical implementation of the information in
:ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.
See there for details on the integration into the frontend.
See the corresponding linked pages on
:ref:`picongpu.profile <install-profile>`
and :ref:`dependencies <install-dependencies>` for more details.


.. _conquering_new_cluster-configuring-frontend:

Configuring the PICMI frontend
------------------------------

The steps above roughly correspond to step 1-3 of the
:ref:`Quick Start <python_package/quickstart:Quick Start>`.
Accordingly, you can pick up from
:ref:`Step 4 <python_package/quickstart:Step 4: Download the Minimal Example>`
once you've worked through the steps above.
You can also try to skip :ref:`manual installation of dependencies <conquering_new_cluster-manual-installation>`
and use :ref:`Step 3 <python_package/quickstart:Step 3: Install the Dependencies>`.


In order for our frontend to use your custom profile and tpl,
please write the following content into a ``picongpurc.toml`` file
and put it
:ref:`in a location where it's picked up by the frontend <configuring_env_toml_file>`.

.. literalinclude:: ../snippets/conquering_new_cluster/picongpurc.toml
   :language: toml

This will use your custom templates in all the following
and will submit runs via slurm (adjust to your system as necessary).


.. _conquering_new_cluster-contributing:

Contributing back
-----------------

A working setup is worth upstreaming so that the next user does not repeat the
work.
A complete contribution adds the following to ``etc/picongpu/<system>/``:

* a ``<system>_picongpu.profile.example``
  (see :ref:`picongpu.profile <install-profile>`),
* one ``*.tpl`` per queue (see the batch templates above),
* a ``dependencies_autoinstall.sh`` if the system does not provide all
  dependencies as modules.

The installer must source ``$PIC_PROFILE`` and guard each install with the
matching ``*_ROOT`` directory check, because those guards are what
``pic-deps check`` verifies.
Open the pull request as described in
`CONTRIBUTING.md <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/CONTRIBUTING.md>`__.
A profile without a working installer is still useful; note in it which
dependencies the user has to provide.

To try an installer before upstreaming, point ``pic-deps`` at it with the
``pic_deps_script`` runtime configuration parameter:

.. literalinclude:: ../snippets/conquering_new_cluster/pic_deps_custom_script.toml
   :language: toml
   :start-after: BEGIN-PIC-DEPS-CUSTOM-SCRIPT
   :end-before: END-PIC-DEPS-CUSTOM-SCRIPT

or pass the same path on the command line.
``--script`` takes precedence over the rc parameter and applies to a single
invocation:

.. literalinclude:: ../snippets/conquering_new_cluster/pic_deps_custom_script.sh
   :language: bash
   :start-after: BEGIN-PIC-DEPS-CUSTOM-SCRIPT
   :end-before: END-PIC-DEPS-CUSTOM-SCRIPT

Like a preset installer, the custom script runs against the profile rendered
from your runtime configuration (``pic-deps`` exports its path as
``$PIC_PROFILE``) and owns the same ``*_ROOT`` guards.
Without a custom script, ``pic-deps`` falls back to the selected preset's
installer. Once the script has stabilised, ship it with the preset so that no
override is needed.
