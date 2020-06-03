.. _usage-python-utils:

Memory Calculator
-----------------

To aid you in the planning and setup of your simulation PIConGPU provides python tools for educated guesses on simulation parameters.
They can be found under ``lib/python/picongpu/utils``.

:ref:`Calculate the memory requirement per device <usage-workflows-memoryPerDevice>`.

    .. code:: python
    
        from picongpu.utils import MemoryCalculator

.. autoclass:: picongpu.utils.memory_calculator.MemoryCalculator
   :members:
   :private-members:
