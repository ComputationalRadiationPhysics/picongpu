.. _install-path:

.. sectionauthor:: Axel Huebl

Installation
============

Installing PIConGPU means :ref:`installing C++ libraries <install-dependencies>` that PIConGPU depends on and :ref:`setting environment variables <install-profile>` to find those dependencies.
The first part is usually the job of a system administrator while the second part needs to be configured on the user-side.

Depending on your experience, role, computing environment and expectations for optimal hardware utilization, you have several ways to install and select PIConGPU's dependencies.
Choose your favorite *install and environment management method* below, young padavan, and follow the corresponding sections of the next chapters.

Ways to Install
---------------

Build from Source
^^^^^^^^^^^^^^^^^

You choose a supported C++ compiler and configure, compile and install all missing dependencies from source.
You are responsible to manage the right versions and configurations.
Performance can be near-ideal if architecture is choosen correctly (and/or if build directly on your hardware).
You then set environment variables to find those installs.

Spack
^^^^^

[Spack]_ is a flexible package manager for HPC systems that can organize versions and dependencies for you.
It can be configured once for your hardware architecture to create optimally tuned binaries and provides module file support (e.g. [Lmod]_).
Those auto-build modules manage your environment variables and allow easy switching between versions, configurations and compilers.

Conda
^^^^^

We currently do not have an official conda install (yet).
Due to pre-build binaries, performance will be sub-ideal and HPC cluster support (e.g. MPI) might be very limited.
Useful for small desktop or single-node runs.

Nvidia-Docker
^^^^^^^^^^^^^

Not yet officially supported but we already provide a `dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_ to get started.
Performance might be sub-ideal if the image is not build for the specific local hardware again.
Useful for small desktop or single-node runs.

Compute Environments
--------------------

HPC Cluster
^^^^^^^^^^^

SysAdmin
""""""""

- use [Spack]_ and auto-build modules, ideally via [Lmod]_
- or build from source, manage binary and version incompatibilities and provide modules

User
""""

As a user, you ideally start with a configured compiler and MPI version for your HPC system (at least).
Those and further dependencies can be set up by:

- loading modules (e.g. via [Lmod]_)

or self-adding them:

- build from source
- or use [Spack]_

Desktop
^^^^^^^

Root/Admin
""""""""""

Use your package manager to install drivers and core dependencies, e.g. via `apt-get install` as far as possible.
Build furhter dependencies from source.

Alternately, use [Spack]_ for all dependencies.

User
""""

If drivers are already installed:

- use [Spack]_
- or use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- or build from source

Cloud
^^^^^

For single nodes, essentially the same as working via SSH on any other machine.
We did not investigate deeper into multi-node cloud setups yet.

AWS
"""

- use [Spack]_
- or use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- or build from source

Google Cloud
""""""""""""

- use [Spack]_
- or use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- or build from source

References
----------

.. [Spack]
        T. Gamblin and contributors.
        *A flexible package manager that supports multiple versions, configurations, platforms, and compilers*,
        `DOI:10.1145/2807591.2807623 <https://dx.doi.org/10.1145/2807591.2807623>`_,
        https://github.com/LLNL/spack

.. [Lmod]
        R. McLay and contributors.
        *Lmod: An Environment Module System based on Lua, Reads TCL Modules, Supports a Software Hierarchy*,
        https://github.com/TACC/Lmod

.. [nvidia-docker]
        Nvidia Corporation and contributors.
        *Build and run Docker containers leveraging NVIDIA GPUs*,
        https://github.com/NVIDIA/nvidia-docker
