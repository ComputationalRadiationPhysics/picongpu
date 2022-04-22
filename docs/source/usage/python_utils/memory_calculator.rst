.. _usage-python-utils-memory-calculator:

Memory Calculator
-----------------

To aid you in the planning and setup of your simulation PIConGPU provides python tools for educated guesses on simulation parameters.
They can be found under ``lib/python/picongpu/extra/utils``.

.. note::

   The utils have been moved to the `picongpu.extra` submodule.

:ref:`Calculate the memory requirement per device <usage-workflows-memoryPerDevice>`.

    .. code:: python
    
        from picongpu.extra.utils import MemoryCalculator

.. autoclass:: picongpu.extra.utils.memory_calculator.MemoryCalculator
   :members:
   :private-members:
