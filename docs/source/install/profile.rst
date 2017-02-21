picongpu.profile
================

Use a `picongpu.profile` file to set up your environment without colliding with other software.
Ideally, store that file directly in your `$HOME/` and source it when after connecting to the machine the first time.

.. code-block:: bash

   . $HOME/picongpu.profile

We listed some example `picongpu.profile` files below which can be used to set up PIConGPU's dependencies on various HPC systems.

Hypnos (HZDR)
-------------

.. literalinclude:: submit/hypnos-hzdr/picongpu.profile.example
   :language: bash
   :linenos:

Titan (ORNL)
------------

.. literalinclude:: submit/titan-ornl/picongpu.profile.example
   :language: bash
   :linenos:

Piz Daint (CSCS)
----------------

.. literalinclude:: submit/pizdaint-cscs/picongpu.profile.example
   :language: bash
   :linenos:

Taurus (TU Dresden)
-------------------

.. literalinclude:: submit/taurus-tud/picongpu.profile.example
   :language: bash
   :linenos:

Lawrencium (LBNL)
-----------------

.. literalinclude:: submit/lawrencium-lbnl/picongpu.profile.example
   :language: bash
   :linenos:

Judge (FZJ)
-----------

(example missing)
