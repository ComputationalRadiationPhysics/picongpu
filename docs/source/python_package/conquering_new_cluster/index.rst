.. _conquering_new_cluster:

Conquering a New Cluster
========================

.. sectionauthor:: Julian J. Lenz

For any HPC system our development team has access to,
we curate a library of presets and an installation tool for dependencies of PIConGPU.
Under many circumstances,
the information in [Quick Start](Quick Start) will get you to a running simulation
without any manual installation.

This section collects the steps for getting PIConGPU running
on a new HPC system that has no preset
and/or is not covered by our dependency installation tool.
It covers installing the dependencies, preparing the environment and integrating this into our PICMI frontend.
Further details on manual compilation are found in [the compilation instructions](keep this in Legacy C++).

.. toctree::
   :maxdepth: 1

   integration
   dependencies
   profile
