.. _conquering_new_cluster-integration:

Integration into the PICMI frontend
===================================

.. sectionauthor:: Julian J. Lenz

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
