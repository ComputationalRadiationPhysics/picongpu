.. _install-profile:

.. seealso::

   You need to have all :ref:`dependencies installed <install-dependencies>` to complete this chapter.

picongpu.profile
================

.. sectionauthor:: Axel Huebl

Use a ``picongpu.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   . $HOME/picongpu.profile

We listed some example ``picongpu.profile`` files below which can be used to set up PIConGPU's dependencies on various HPC systems.

Hypnos (HZDR)
-------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

.. literalinclude:: submit/hypnos-hzdr/picongpu.profile.example
   :language: bash

Titan (ORNL)
------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libSplash, libpng and PNGwriter <install-dependencies>` manually.

.. literalinclude:: submit/titan-ornl/picongpu.profile.example
   :language: bash

Piz Daint (CSCS)
----------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, PNGwriter and ADIOS <install-dependencies>` manually.

.. literalinclude:: submit/pizdaint-cscs/picongpu.profile.example
   :language: bash

Taurus (TU Dresden)
-------------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and libSplash <install-dependencies>` manually.

.. literalinclude:: submit/taurus-tud/picongpu.profile.example
   :language: bash

Lawrencium (LBNL)
-----------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, PNGwriter and libSplash <install-dependencies>` manually.
Additionally, you need to make the ``rsync`` command available as written below.

.. literalinclude:: submit/lawrencium-lbnl/picongpu.profile.example
   :language: bash

Judge (FZJ)
-----------

(example missing)
