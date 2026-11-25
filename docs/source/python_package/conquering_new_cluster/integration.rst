.. _conquering_new_cluster-process:

Integration into the PICMI frontend
===================================

.. sectionauthor:: Julian J. Lenz

As described in detail in [Foundations](...),
the Python frontend is basically an orchestrator composing different pieces of information
into the full workflow of compiling and running your PIConGPU setup tailored to the system at hand.
On curated systems, these pieces of information are provided by the preset.
On a new system, we need to provide:
1. Environment information in the form of a [picongpu.profile](...)
2. An installation of [dependencies](...) that are not provided by the system itself

This page describes how to inform the orchestrator about these components.
This roughly corresponds to step 1 and 2 of the [Quick Start](...).
Accordingly, you can pick up from [Step 3](...) once you've worked through the steps below.

This page is basically the practical implementation of the information in [Configuring your environment](...).
See there for details on the integration into the frontend.
See the corresponding linked pages on [picongpu.profile](...) and [dependencies](...)
for more details.


Authoring a custom profile
--------------------------

The following process is suggested for arriving at a custom profile.
0. Read [picongpu.profile](...) in order to gain a basic understanding of the following edits.
1. Look through [etc/picongpu](...) and look for a system similar to the one you're working with
   (e.g. in terms of batch system, hardware, ...).
2. Copy `*_picongpu.profile.example` as a starting point.
   We'll call this `custom.profile` in the following.
3. Figure out which of the [dependencies](...) are provided by the system,
   typically in form of a module system.
   Commands like `module avail` and your system's documentation will probably be helpful.
4. Edit your copy of the profile to load all your system-provided dependencies
   and take a note of the unfulfilled dependencies.
5. Check the remaining content and adjust system-specific details.
6. Make sure it exports `PIC_LIBS={{{pic_libs}}}` and the corresponding dependency root environment variables
   (e.g. `BOOST_ROOT=$PIC_LIBS/...`, ...) making those dependencies available.
7. Replace the line `export PICSRC=...` with `export PICSCR={{{pic_src_path}}}`


Configuring the PICMI frontend
------------------------------

In order for our frontend to use your custom profile,
please write the following content into a `picongpurc.toml` file
and put it [in a location where it's picked up by the frontend](...).

```
profile_template_path="/path/to/custom.profile"
submit_system="sbatch"
pic_libs="/path/to/where/dependencies/shall/be/installed"
```

This will use your hardcoded profile in all the following
and will submit runs via slurm (adjust to your system as necessary).

You can now pick up from [Step 3 of the Quick Start Guide](...) to install your dependencies.
