.. _usage-workflows-memoryPerDevice:

Calculating the Memory Requirement per Device
---------------------------------------------

.. sectionauthor:: Marco Garten

The planning of simulations for realistically sized problems requires a careful estimation of memory usage and is often a trade-off between resolution of the plasma, overall box size and the available resources.
The file :ref:`memory_calculator.py <usage-python-utils>` contains a class for this purpose.

The following paragraph shows the use of the ``MemoryCalculator`` for the ``4.cfg`` setup of the :ref:`FoilLCT example <usage-examples-foilLCT>` example.

.. literalinclude:: ./memoryPerDevice.py
    :language: python3
    :lines: 11,26-

This will give the following output:

.. program-output:: bash -c "PYTHONPATH=$(pwd)/../../lib/python:$PYTHONPATH ./usage/workflows/memoryPerDevice.py"

If you have a machine or cluster node with NVIDIA GPUs you can find out the available memory size by typing ``nvidia-smi`` on a shell.