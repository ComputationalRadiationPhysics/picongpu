.. _usage-python-utils:

Utilities
---------

Python Preprocessing
====================

To aid you in the planning and setup of your simulation PIConGPU provides python tools for educated guesses on simulation parameters.
They can be found under ``lib/python/picongpu/utils``.

It is our hope that this list will grow alongside with our user base to make using PIConGPU easier for everyone.

1. :ref:`Calculate the memory requirement per device <usage-workflows-memoryPerDevice>` with ``memory_calculator.py``

memory_calculator.py
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: picongpu.utils.memory_calculator.MemoryCalculator
   :members:
   :private-members:

