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

A selection of available profiles is presented below, after some general notes on using CPUs.
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
This approach is used, for example, in the ``defq`` partition of Hemera as shown below.

The properties of OpenMP parallelization, such as number of threads used, are controlled via OpenMP environment variables.
In particular, the number of OpenMP threads to be used (per MPI rank) can be set via ``$OMP_NUM_THREADS``.
Beware that task launch wrappers used on your system may effectively override this setting.
Particularly, a few systems require running PIConGPU with ``mpirun --bind-to none`` in order to properly use all CPU cores.

For setting thread affinity, we provide a helper wrapper ``cpuNumaStarter.sh`` that should be applicable to most systems.

Your Workstation
----------------

This is a very basic ``picongpu.profile`` enabling compilation on CPUs by setting the OpenMP backend, declaring commonly required directories,
and providing default parameters for :ref:`TBG <usage-tbg>`.

.. literalinclude:: profiles/bash/bash_picongpu.profile.example
   :language: bash

Crusher (ORNL)
--------------

**System overview:** `link <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#system-overview>`__

**Production directory:** usually ``$PROJWORK/$proj/`` (`link <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#data-and-storage>`__).
Note that ``$HOME`` is mounted on compute nodes as read-only.

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and openPMD <install-dependencies>` manually.

MI250X GPUs using hipcc (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/crusher-ornl/batch_hipcc_picongpu.profile.example
   :language: bash

MI250X GPUs using craycc
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/crusher-ornl/batch_craycc_picongpu.profile.example
  :language: bash

Hemera (HZDR)
-------------

**System overview:** `link (internal) <https://www.hzdr.de/db/Cms?pOid=29813>`__

**User guide:** *None*

**Production directory:** ``/bigdata/hplsim/`` with ``external/``, ``scratch/``, ``development/`` and ``production/``

Profile for HZDR's home cluster hemera.
Sets up software environment, i.e. providing libraries to satisfy PIConGPU's dependencies, by loading modules,
setting common paths and options, as well as defining the ``getDevice()`` and ``getNode()`` aliases.
The latter are shorthands to request resources for an interactive session from the batch system.
Together with the `-s bash` option of :ref:`TBG <usage-tbg>`, these allow to run PIConGPU interactively on an HPC system.


For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: defq (2x Intel Xeon Gold 6148, 20 Cores + 20 HyperThreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/defq_picongpu.profile.example
   :language: bash

Queue: gpu (4x NVIDIA P100 16GB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/gpu_picongpu.profile.example
   :language: bash

Queue: fwkt_v100 (4x NVIDIA V100 32GB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/fwkt_v100_picongpu.profile.example
   :language: bash

Summit (ORNL)
-------------

**System overview:** `link <https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/>`__

**User guide:** `link <https://www.olcf.ornl.gov/for-users/system-user-guides/summit/>`__

**Production directory:** usually ``$PROJWORK/$proj/`` (`link <https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/#file-systems>`__).
Note that ``$HOME`` is mounted on compute nodes as read-only.

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter <install-dependencies>` manually.

V100 GPUs (recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/summit-ornl/gpu_picongpu.profile.example
   :language: bash

Piz Daint (CSCS)
----------------

**System overview:** `link <https://www.cscs.ch/computers/piz-daint/>`__

**User guide:** `link <https://user.cscs.ch/>`__

**Production directory:** ``$SCRATCH`` (`link <https://user.cscs.ch/storage/file_systems/>`__).

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, libpng, PNGwriter and ADIOS2 <install-dependencies>` manually.

.. note::

   The MPI libraries are lacking Fortran bindings (which we do not need anyway).
   During the install of ADIOS, make sure to add to ``configure`` the ``--disable-fortran`` flag.

.. note::

   Please find a `Piz Daint quick start from August 2018 here <https://gist.github.com/ax3l/68cb4caa597df3def9b01640959ea56b>`_.

.. literalinclude:: profiles/pizdaint-cscs/picongpu.profile.example
   :language: bash

Taurus (TU Dresden)
-------------------

**System overview:** `link <https://tu-dresden.de/zih/hochleistungsrechnen/hpc>`__

**User guide:** `link <https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/SystemTaurus>`__

**Production directory:** ``/scratch/$USER/`` and ``/scratch/$proj/``

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter <install-dependencies>` manually.

Queue: gpu2 (Nvidia K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/taurus-tud/k80_picongpu.profile.example
   :language: bash

Queue: ml (NVIDIA V100 GPUs on Power9 nodes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this profile, you additionally need to compile and install everything for the power9-architecture including your own :ref:`boost <install-dependencies>`, :ref:`HDF5 <install-dependencies>`, c-blosc and :ref:`ADIOS <install-dependencies>`.

.. note::

   Please find a `Taurus ml quick start here <https://gist.github.com/steindev/cc02eae81f465833afa27fc8880f3473>`_.

.. note::
   
   You need to compile the libraries and PIConGPU on an ``ml`` node since
   only nodes in the ``ml`` queue are Power9 systems.

.. literalinclude:: profiles/taurus-tud/V100_picongpu.profile.example
   :language: bash

Cori (NERSC)
------------

**System overview:** `link <https://www.nersc.gov/users/computational-systems/cori/configuration/>`__

**User guide:** `link <https://docs.nersc.gov/>`__

**Production directory:** ``$SCRATCH`` (`link <https://www.nersc.gov/users/storage-and-file-systems/>`__).

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter <install-dependencies>` manually.

Queue: dgx (DGX - A100)
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/cori-nersc/a100_picongpu.profile.example
   :language: bash

Draco (MPCDF)
-------------

**System overview:** `link <https://www.mpcdf.mpg.de/services/computing/draco/about-the-system>`__

**User guide:** `link <https://www.mpcdf.mpg.de/services/computing/draco>`__

**Production directory:** ``/ptmp/$USER/``

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libpng and PNGwriter <install-dependencies>` manually.

.. literalinclude:: profiles/draco-mpcdf/picongpu.profile.example
   :language: bash

D.A.V.I.D.E (CINECA)
--------------------

**System overview:** `link <http://www.hpc.cineca.it/content/davide>`__

**User guide:** `link <https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+D.A.V.I.D.E.+UserGuide>`__

**Production directory:** ``$CINECA_SCRATCH/`` (`link <https://wiki.u-gov.it/confluence/display/SCAIUS/UG2.4%3A+Data+storage+and+FileSystems>`__)

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: dvd_usr_prod (Nvidia P100 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/davide-cineca/gpu_picongpu.profile.example
   :language: bash

JURECA (JSC)
------------

**System overview:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html>`__

**User guide:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/UserInfo/UserInfo_node.html>`__

**Production directory:** ``$SCRATCH`` (`link <http://www.fz-juelich.de/SharedDocs/FAQs/IAS/JSC/EN/JUST/FAQ_00_File_systems.html?nn=1297148>`__)

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and openPMD <install-dependencies>`, for the gpus partition also :ref:`Boost and HDF5 <install-dependencies>`, manually.

Queue: batch (2 x Intel Xeon E5-2680 v3 CPUs, 12 Cores + 12 Hyperthreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/jureca-jsc/batch_picongpu.profile.example
   :language: bash

Queue: gpus (2 x Nvidia Tesla K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/jureca-jsc/gpus_picongpu.profile.example
   :language: bash

Queue: booster (Intel Xeon Phi 7250-F, 68 cores + Hyperthreads)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/jureca-jsc/booster_picongpu.profile.example
   :language: bash

JUWELS (JSC)
------------

**System overview:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html>`__

**User guide:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/UserInfo/UserInfo_node.html>`__

**Production directory:** ``$SCRATCH`` (`link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/FAQ/juwels_FAQ_node.html#faq1495160>`__)

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and openPMD <install-dependencies>`, for the gpus partition also :ref:`Boost and HDF5 <install-dependencies>`, manually.

Queue: batch (2 x Intel Xeon Platinum 8168 CPUs, 24 Cores + 24 Hyperthreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/juwels-jsc/batch_picongpu.profile.example
   :language: bash

Queue: gpus (4 x Nvidia V100 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/juwels-jsc/gpus_picongpu.profile.example
   :language: bash

ARIS (GRNET)
------------

**System overview:** `link <http://doc.aris.grnet.gr/>`__

**User guide:** `link <http://doc.aris.grnet.gr/environment/>`__

**Production directory:** ``$WORKDIR`` (`link <http://doc.aris.grnet.gr/system/storage/>`__)

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>`.

Queue: gpu (2 x NVIDIA Tesla k40m GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/aris-grnet/gpu_picongpu.profile.example
   :language: bash

Ascent (ORNL)
-------------

**System overview and user guide:** `link <https://docs.olcf.ornl.gov/systems/ascent_user_guide.html#system-overview/>`__

**Production directory:** usually ``$PROJWORK/$proj/`` (as on summit `link <https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/#file-systems>`__).

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`openPMD-api and PNGwriter <install-dependencies>` manually or use pre-installed libraries in the shared project directory.

V100 GPUs (recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/ascent-ornl/gpu_picongpu.profile.example
   :language: bash
