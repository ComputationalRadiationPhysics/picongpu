.. _install-path:

.. sectionauthor:: Axel Huebl

Installation
============

Installing PIConGPU means :ref:`installing C++ libraries <install-dependencies>` that PIConGPU depends on and :ref:`setting environment variables <install-profile>` to find those dependencies.
The first part is usually the job of a system administrator while the second part needs to be configured on the user-side.

Depending on your experience, role, computing environment and expectations optimal hardware utilization, you have several ways to install and select PIConGPU's dependencies.
Choose your favorite *install and environment management method* below, young padavan, and follow the corresponding sections of the next chapters.

HPC Cluster
-----------

SysAdmin
^^^^^^^^

- use [spack]_
- build from source and provide modules, ideally via [Lmod]_

User
^^^^

- load modules (e.g. via [Lmod]_)
- missing: build from source
- use [spack]_

Desktop
-------

Root/Admin
^^^^^^^^^^

- `apt-get install` what is possible
- use [spack]_
- build from source

User
^^^^

- use [spack]_
- use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- build from source

Cloud
-----

AWS
^^^

- use [spack]_
- use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- build from source

Google Cloud
^^^^^^^^^^^^

- use [spack]_
- use [nvidia-docker]_ (`dockerfile <https://github.com/ComputationalRadiationPhysics/picongpu/issues/829>`_)
- build from source

References
----------

.. [spack]
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
