.. _install-profile:

.. sectionauthor:: Axel Huebl

.. seealso::

   You need to have all :ref:`dependencies installed <install-dependencies>` to complete this chapter.

picongpu.profile
================

Use a ``picongpu.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   . $HOME/picongpu.profile

We listed some example `picongpu.profile` files below which can be used to set up PIConGPU's dependencies on various HPC systems.

Hypnos (HZDR)
-------------

.. literalinclude:: submit/hypnos-hzdr/picongpu.profile.example
   :language: bash

Titan (ORNL)
------------

.. literalinclude:: submit/titan-ornl/picongpu.profile.example
   :language: bash

Piz Daint (CSCS)
----------------

.. literalinclude:: submit/pizdaint-cscs/picongpu.profile.example
   :language: bash

Taurus (TU Dresden)
-------------------

.. literalinclude:: submit/taurus-tud/picongpu.profile.example
   :language: bash

Lawrencium (LBNL)
-----------------

.. literalinclude:: submit/lawrencium-lbnl/picongpu.profile.example
   :language: bash

Judge (FZJ)
-----------

(example missing)
