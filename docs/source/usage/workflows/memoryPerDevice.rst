.. _usage-workflows-memoryPerDevice:

Calculating the Memory Requirement per Device
---------------------------------------------

.. sectionauthor:: Marco Garten

The planning of simulations for realistically sized problems requires a careful estimation of memory usage and is often a trade-off between resolution of the plasma, overall box size and the available resources.
The file :ref:`memory_calculator.py <usage-python-utils-memory-calculator>` contains a class for this purpose.

The following paragraph shows the use of the ``MemoryCalculator`` for the ``4.cfg`` setup of the :ref:`FoilLCT example <usage-examples-foilLCT>` example.

It is an estimate for how much memory is used per device if the whole
target would be fully ionized but does not move much. Of course, the real
memory usage depends on the case and the dynamics inside the simulation.
We calculate the memory of just one device per row of GPUs in laser
propagation direction. We hereby assume that particles are distributed
equally in the transverse direction like it is set up in the FoilLCT example.

We encourage to try out this script with different settings, to see
how they influence the distribution of the total memory requirement
between devices.

.. literalinclude:: ./memoryPerDevice.py
    :language: python3
    :lines: 11,12,27-

This will give the following output:

.. program-output:: bash -c "PYTHONPATH=$(pwd)/../../lib/python:$PYTHONPATH ./usage/workflows/memoryPerDevice.py"

If you have a machine or cluster node with NVIDIA GPUs you can find out the available memory size by typing ``nvidia-smi`` on a shell.
