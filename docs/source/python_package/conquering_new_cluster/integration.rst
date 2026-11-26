.. _conquering_new_cluster-integration:

Integration into the PICMI frontend
===================================

.. sectionauthor:: Julian Lenz

As described in detail in
:ref:`Foundations <python_package/foundations/index:Foundations>`,
the Python frontend is basically an orchestrator composing different pieces of information
into the full workflow of compiling and running your PIConGPU setup tailored to the system at hand.
On curated systems, these pieces of information are provided by the preset.
On a new system, we need to provide:

#. Environment information in the form of a
   :ref:`picongpu.profile <install-profile>`
#. An installation of :ref:`dependencies <install-dependencies>`
   that are not provided by the system itself

This page describes how to inform the orchestrator about these components.
This roughly corresponds to step 1 and 2 of the
:ref:`Quick Start <python_package/quickstart:Quick Start>`.
Accordingly, you can pick up from
:ref:`Step 3 <python_package/quickstart:Step 3: Install the Dependencies>`
once you've worked through the steps below.

This page is basically the practical implementation of the information in
:ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.
See there for details on the integration into the frontend.
See the corresponding linked pages on
:ref:`picongpu.profile <install-profile>`
and :ref:`dependencies <install-dependencies>` for more details.


Authoring a custom profile
--------------------------

The following process is suggested for arriving at a custom profile.

0. Read :ref:`picongpu.profile <install-profile>` in order to gain a basic
   understanding of the following edits.
1. Look through
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
5. Check the remaining content and adjust system-specific details.
6. Make sure it exports ``PIC_LIBS={{{pic_libs}}}`` and the corresponding
   dependency root environment variables
   (e.g. ``BOOST_ROOT=$PIC_LIBS/...``, ...) making those dependencies
   available.
7. Replace the line ``export PICSRC=...`` with
   ``export PICSRC={{{pic_src_path}}}``


.. _conquering_new_cluster-templates:

Batch templates
---------------

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

The runtime configuration selects the source template with ``tbg_tpl_file``
(``TBG_TPLFILE`` in a profile) and the submit command with ``tbg_submit``.
``tbg`` copies the source template to the run's ``tbg/submit.tpl`` and writes
the resolved batch script to ``tbg/submit.start``; ``tbg/submit.cfg`` records
the configuration.
``submit.tpl`` is thus a record of what produced the job, not a file to edit.

A new cluster needs its own template when the scheduler directives, the
launcher or the resource accounting differ from every shipped system.
Copy the closest system template (or a generic launcher), adjust the
``.TBG_*`` computations and the launch line, and set ``tbg_tpl_file`` in your
``.picongpurc.toml`` as shown in
:ref:`Configuring the PICMI frontend <conquering_new_cluster-configuring-frontend>`.
Small adjustments can often be made with ``overwrite_vars`` of
``simulation.run()`` instead of editing the template
(see :ref:`the tbg flags <tbg-flags>`).
Relative template paths are interpreted relative to the current working
directory.


.. _conquering_new_cluster-mpi:

MPI environments
----------------

MPI is the most common source of "works on one cluster, fails on the next"
problems.
There is no portable configuration; treat the following as checks against the
cluster's own documentation.

* **Compiler and MPI ABI.**
  Build PIConGPU and all dependencies with the same compiler family and,
  ideally, version, and against the same MPI installation used at run time.
  Use the compiler wrappers of that MPI (``mpicc``/``mpicxx`` or the site's
  ``CC``/``CXX``). Mixing an MPI ABI with another compiler or with libraries
  built against a different MPI yields link errors or crashes at run time.
* **Launcher.**
  Slurm systems often use ``srun``, others use ``mpiexec`` or ``mpirun``;
  some sites provide wrappers that map one to the other, others do not.
  The launch line in the template must use the launcher the system supports.
  The generic templates show ``mpiexec`` and ``mpirun`` variants.
* **Fabrics.**
  Interconnect settings such as ``UCX_TLS``, ``UCX_NET_DEVICES``,
  ``OFI_PROVIDER`` or ``--mca`` options are site- and job-specific.
  Take them from the cluster documentation or a working job script.
* **GPU-aware MPI.**
  GPU runs need an MPI built for the vendor's stack (CUDA- or ROCm-aware).
  A non-aware MPI may still work with host-staged transfers but is slower and
  may need explicit settings. Enable GPU awareness consistently in the build
  and on the launch line.
* **CPU binding.**
  Thread and rank placement is controlled by launch options such as
  ``--bind-to``, ``--cpu-bind`` or ``mpirun --bind-to none`` and by
  ``OMP_NUM_THREADS``. Launchers and wrappers may override what the profile
  exports. See the :ref:`CPU notes in the profile documentation
  <install-profile>` for a starting point.


.. _conquering_new_cluster-configuring-frontend:

Configuring the PICMI frontend
------------------------------

In order for our frontend to use your custom profile,
please write the following content into a ``picongpurc.toml`` file
and put it
:ref:`in a location where it's picked up by the frontend <configuring_env_toml_file>`.

.. literalinclude:: ../snippets/conquering_new_cluster/picongpurc.toml
   :language: toml

This will use your hardcoded profile in all the following
and will submit runs via slurm (adjust to your system as necessary).

You can now pick up from
:ref:`Step 3 of the Quick Start Guide <python_package/quickstart:Step 3: Install the Dependencies>`
to install your dependencies.


.. _conquering_new_cluster-manual-installation:

Installing dependencies manually
--------------------------------

``pic-deps`` is a best-effort convenience.
It assumes a usable shell toolchain, a compatible MPI, network access and a
preset (or script) that knows where to put the libraries.
On a new system it is unlikely to "just work"; treat its output as a starting
point and always verify it with ``pic-deps check``.

The installers under ``etc/picongpu/bash/``
(``dependencies_autoinstall.sh``, ``dependencies_autoinstall_spack.sh`` and
``dependencies_autoinstall_conda.sh``) are complete module-less recipes and
the best starting point.
Copy one next to your preset, or point ``pic-deps`` at it with
:ref:`a custom script <conquering_new_cluster-contributing>`, and adapt the
package versions and download locations to the system.

If that is not sufficient, install the dependencies manually:
follow the cluster's documentation for the available modules and recommended
toolchains, load the system-provided modules for the compiler, CUDA/ROCm, MPI,
HDF5, Boost and the remaining libraries (using one consistent compiler/MPI
combination throughout), build what is missing from source into a writable
prefix, and export the corresponding ``*_ROOT`` variables.
The full list, versions and package-manager hints are in
:ref:`dependencies <install-dependencies>`; wiring the resulting variables
into the environment is described in
:ref:`picongpu.profile <install-profile>`.
Compile once to verify the environment before submitting a large run.


.. _conquering_new_cluster-contributing:

Contributing back
-----------------

A working setup is worth upstreaming so that the next user does not repeat the
work.
A complete contribution adds the following to ``etc/picongpu/<system>/``:

* a ``<system>_picongpu.profile.example``
  (see :ref:`picongpu.profile <install-profile>`),
* one ``*.tpl`` per queue (see the batch templates above),
* a ``dependencies_autoinstall*.sh`` if the system does not provide all
  dependencies as modules.
  Start from ``etc/picongpu/bash/dependencies_autoinstall.sh`` or the
  ``_spack``/``_conda`` variant.

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
