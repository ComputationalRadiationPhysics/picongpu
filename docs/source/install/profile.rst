.. _install-profile:

.. seealso::

   You need to have all :ref:`dependencies installed <install-dependencies>` to complete this chapter.

picongpu.profile
================

.. sectionauthor:: Axel Huebl, Klaus Steiniger, Sergei Bastrakov

We recommend to use a ``picongpu.profile`` file, located directly in your ``$HOME/`` directory,
to set up the environment within which PIConGPU will run by conviently performing

.. code-block:: bash

   source $HOME/picongpu.profile

on the command line after logging in to a system.
PIConGPU is shipped with a number of ready-to-use profiles for different systems which are located in
``etc/picongpu/<cluster>-<institute>/`` within PIConGPU's main folder.
Have a look into this directory in order to see for which HPC systems profiles are already available.
If you are working on one of these systems, just copy the respective ``*_picongpu.profile.example``
from within this directory into your ``$HOME`` and make the necessary changes, such as e-mail address
or PIConGPU source code location defined by ``$PICSRC``.
If you are working on an HPC system for which no profile is available, feel free to create one and
contribute it to PIConGPU by opening a pull request.

System-specific notes for a few machines are collected below.
Beware, these may not be up-to-date with the latest available software on the respective system,
as we do not have continuous access to all of these.

General Notes on Using CPUs
---------------------------

On CPU systems we strongly recommend using MPI + OpenMP parallelization.
It requires building PIConGPU with the OpenMP 2 backend.
Additionally it is recommended to add an option for target architecture, for example, ``pic-build -b omp2b:znver3`` for AMD Zen3 CPUs.
When building on a compute node or a same-architecture node, one could use ``-b omp2b:native`` instead.
The default value for option ``-b`` can be set with environment variable ``$PIC_BACKEND`` in the profile.

With respect to selecting an optimal MPI + OpenMP configuration please refer to documentation of your system.
As a reasonable default strategy, we recommend running an MPI rank per NUMA node, using 1 or 2 OpenMP threads per core depending on simultaneous multithreading being enabled, and binding threads to cores through affinity settings.
This approach is used, for example, in the ``defq`` partition of Hemera.

The properties of OpenMP parallelization, such as number of threads used, are controlled via OpenMP environment variables.
In particular, the number of OpenMP threads to be used (per MPI rank) can be set via ``$OMP_NUM_THREADS``.
Beware that task launch wrappers used on your system may effectively override this setting.
Particularly, a few systems require running PIConGPU with ``mpirun --bind-to none`` in order to properly use all CPU cores.

For setting thread affinity, we provide a helper wrapper ``cpuNumaStarter.sh`` that should be applicable to most systems.

dev server (HZDR)
-----------------

**information on software setup with spack:**

.. toctree::
   :glob:
   :maxdepth: 1

   ./profiles/bash-devServer-hzdr/README

Capella (ZIH)
-------------

Information on SLURM usage
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :glob:
   :maxdepth: 1

   ./profiles/zih-tud/Slurm_Tutorial

Perlmutter (NERSC)
------------------

General information on system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :glob:
   :maxdepth: 1

   ./profiles/perlmutter-nersc/README
