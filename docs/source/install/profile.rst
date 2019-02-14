.. _install-profile:

.. seealso::

   You need to have all :ref:`dependencies installed <install-dependencies>` to complete this chapter.

picongpu.profile
================

.. sectionauthor:: Axel Huebl

Use a ``picongpu.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   source $HOME/picongpu.profile

We listed some example ``picongpu.profile`` files below which can be used to set up PIConGPU's dependencies on various HPC systems.

Hemera (HZDR)
-------------

**System overview:** `link (internal) <https://www.hzdr.de/db/Cms?pOid=29813>`_

**User guide:** *None*

**Production directory:** ``/bigdata/hplsim/`` with ``external/``, ``scratch/``, ``development/`` and ``production/``

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: defq (2x Intel Xeon Gold 6148, 20 Cores + 20 HyperThreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/defq_picongpu.profile.example
   :language: bash

Queue: gpu (4x NVIDIA P100 16GB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/gpu_picongpu.profile.example
   :language: bash

Hypnos (HZDR)
-------------

**System overview:** `link (internal) <https://www.hzdr.de/db/Cms?pOid=29813>`_

**User guide:** `link (internal) <http://hypnos3/wiki>`_

**Production directory:** ``/bigdata/hplsim/`` with ``external/``, ``scratch/``, ``development/`` and ``production/``

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: laser (AMD Opteron 6276 CPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/laser_picongpu.profile.example
   :language: bash

Queue: k20 (Nvidia K20 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/k20_picongpu.profile.example
   :language: bash

Queue: k80 (Nvidia K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/k80_picongpu.profile.example
   :language: bash

Hydra (HZDR)
-------------

**System overview:** `link (internal) <https://www.hzdr.de/db/Cms?pOid=29813>`_

**User guide:** `link (internal) <http://hypnos3/wiki>`_

**Production directory:** ``/bigdata/hplsim/`` with ``external/``, ``scratch/``, ``development/`` and ``production/``

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

.. literalinclude:: profiles/hydra-hzdr/default_picongpu.profile.example
   :language: bash

Titan (ORNL)
------------

**System overview:** `link <https://www.olcf.ornl.gov/olcf-resources/compute-systems/titan/>`_

**User guide:** `link <https://www.olcf.ornl.gov/for-users/system-user-guides/titan/>`_

**Production directory:** usually ``$PROJWORK/$proj/`` (`link <https://www.olcf.ornl.gov/for-users/system-user-guides/titan/file-systems/>`_).
Note that ``$HOME`` is not mounted on compute nodes, place your ``picongpu.profile`` and auxiliary software in your production directory.

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libSplash, libpng and PNGwriter <install-dependencies>` manually.

K20x GPUs (recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/titan-ornl/gpu_picongpu.profile.example
   :language: bash

AMD Opteron 6274 (Interlagos) CPUs (for experiments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/titan-ornl/cpu_picongpu.profile.example
   :language: bash

Piz Daint (CSCS)
----------------

**System overview:** `link <https://www.cscs.ch/computers/piz-daint/>`_

**User guide:** `link <https://user.cscs.ch/>`_

**Production directory:** ``$SCRATCH`` (`link <https://user.cscs.ch/storage/file_systems/>`_).

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, zlib, libpng, c-blosc, PNGwriter, libSplash and ADIOS <install-dependencies>` manually.

.. note::

   The MPI libraries are lacking Fortran bindings (which we do not need anyway).
   During the install of ADIOS, make sure to add to ``configure`` the ``--disable-fortran`` flag.

.. note::

   Please find a `Piz Daint quick start from August 2018 here <https://gist.github.com/ax3l/68cb4caa597df3def9b01640959ea56b>`_.

.. literalinclude:: profiles/pizdaint-cscs/picongpu.profile.example
   :language: bash

Taurus (TU Dresden)
-------------------

**System overview:** `link <https://tu-dresden.de/zih/hochleistungsrechnen/hpc>`_

**User guide:** `link <https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/SystemTaurus>`_

**Production directory:** ``/scratch/$USER/`` and ``/scratch/$proj/``

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and libSplash <install-dependencies>` manually.

Queue: gpu1 (Nvidia K20x GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/taurus-tud/k20x_picongpu.profile.example
   :language: bash

Queue: gpu2 (Nvidia K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/taurus-tud/k80_picongpu.profile.example
   :language: bash

Queue: knl (Intel  Intel Xeon Phi - Knights Landing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this profile, you additionally need to install your own :ref:`boost <install-dependencies>`.

.. literalinclude:: profiles/taurus-tud/knl_picongpu.profile.example
   :language: bash
   
Queue: ml (NVIDIA V100 GPUs on Power9 nodes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this profile, you additionally need to compile and install everything for the power9-architecture including your own :ref:`boost <install-dependencies>`, :ref:`HDF5 <install-dependencies>`, c-blosc and :ref:`ADIOS <install-dependencies>`.

Install script for `c-blosc`

.. code-block:: bash

   cd $SOURCE_DIR
   git clone -b v1.12.1 https://github.com/Blosc/c-blosc.git \
       $SOURCE_DIR/c-blosc
   mkdir c-blosc-build
   cd c-blosc-build
   cmake -DCMAKE_INSTALL_PREFIX=$BLOSC_ROOT \
       -DPREFER_EXTERNAL_ZLIB=ON \
       $SOURCE_DIR/c-blosc
   make -j4
   make install

.. literalinclude:: profiles/taurus-tud/V100_picongpu.profile.example
   :language: bash

Lawrencium (LBNL)
-----------------

**System overview:** `link <http://scs.lbl.gov/Systems>`_

**User guide:** `link <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`_

**Production directory:** ``/global/scratch/$USER/``

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, PNGwriter and libSplash <install-dependencies>` manually.
Additionally, you need to make the ``rsync`` command available as written below.

.. literalinclude:: profiles/lawrencium-lbnl/picongpu.profile.example
   :language: bash

Draco (MPCDF)
-------------

**System overview:** `link <https://www.mpcdf.mpg.de/services/computing/draco/about-the-system>`_

**User guide:** `link <https://www.mpcdf.mpg.de/services/computing/draco>`_

**Production directory:** ``/ptmp/$USER/``

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libpng, PNGwriter and libSplash <install-dependencies>` manually.

.. literalinclude:: profiles/draco-mpcdf/picongpu.profile.example
   :language: bash

D.A.V.I.D.E (CINECA)
--------------------

**System overview:** `link <http://www.hpc.cineca.it/content/davide>`_

**User guide:** `link <https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+D.A.V.I.D.E.+UserGuide>`_

**Production directory:** ``$CINECA_SCRATCH/`` (`link <https://wiki.u-gov.it/confluence/display/SCAIUS/UG2.4%3A+Data+storage+and+FileSystems>`_)

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: dvd_usr_prod (Nvidia P100 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/davide-cineca/gpu_picongpu.profile.example
   :language: bash

JURECA (JSC)
------------

**System overview:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html>`_

**User guide:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/UserInfo/UserInfo_node.html>`_

**Production directory:** ``$SCRATCH`` (`link <http://www.fz-juelich.de/SharedDocs/FAQs/IAS/JSC/EN/JUST/FAQ_00_File_systems.html?nn=1297148>`_)

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter, c-blosc, adios and libSplash <install-dependencies>`, for the gpus partition also :ref:`Boost and HDF5 <install-dependencies>`, manually.

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

**System overview:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html>`_

**User guide:** `link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/UserInfo/UserInfo_node.html>`_

**Production directory:** ``$SCRATCH`` (`link <http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/FAQ/juwels_FAQ_node.html#faq1495160>`_)

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter, c-blosc, adios and libSplash <install-dependencies>`, for the gpus partition also :ref:`Boost and HDF5 <install-dependencies>`, manually.

Queue: batch (2 x Intel Xeon Platinum 8168 CPUs, 24 Cores + 24 Hyperthreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/juwels-jsc/batch_picongpu.profile.example
   :language: bash

Queue: gpus (4 x Nvidia V100 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/juwels-jsc/gpus_picongpu.profile.example
   :language: bash
