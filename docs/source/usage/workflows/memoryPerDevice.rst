.. _usage-workflows-memoryPerDevice:

Calculating the Memory Requirement per Device
---------------------------------------------

.. sectionauthor:: Marco Garten

The planning of simulations for realistically sized problems requires a careful estimation of memory usage and is often a trade-off between resolution of the plasma, overall box size and the available resources.
The file :ref:`memory_calculator.py <usage-python-utils-memory-calculator>` contains a class for this purpose.

The following example script demonstrates it's use:

.. literalinclude:: ./memoryPerDevice.py
    :language: python3

We encourage to try out this script with different settings, to see
how they influence the distribution of the total memory requirement
between devices.

This will give the following output:

.. program-output:: bash -c "PYTHONPATH=$(pwd)/../../lib/python:$PYTHONPATH /usr/bin/env python -m pip install -e ../../lib/python &> /dev/null; PYTHONPATH=$(pwd)/../../lib/python:$PYTHONPATH ./usage/workflows/memoryPerDevice.py"

If you have a machine or cluster node with NVIDIA GPUs you can find out the available memory size by typing ``nvidia-smi`` on a shell.
