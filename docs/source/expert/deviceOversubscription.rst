.. _expert-deviceOversubscription:

Device Oversubscription
=======================

.. moduleauthor:: René Widera

By default the strategy to execute PIConGPU is that one MPI rank is using a single device e.g. a GPU.
In some situation it could be beneficial to use multiple MPI ranks per device to get a better load balancing
or better overlap communications with computation.

Usage
-----

Follow :ref:`the description to pass command line parameter to PIConGPU <usage-cfg>`.
PIConGPU provides the command line parameter ``--numRanksPerDevice`` or short ``-r`` to allow sharing a compute device
between multiple MPI ranks.
If you change the default value ``1`` to ``2`` PIConGPU is supporting two MPI processes per device.

.. note::
    Using device oversubscription will limit the maximal memory footprint per PIConGPU MPI rank on the device too
    ``<total available memory on device>/<number of ranks per device>``.

NVIDIA
------

Compute Mode
^^^^^^^^^^^^

On NVIDIA GPUs there are different point which can influence the oversubscription of a device/GPU.
`NVIDIA Compute Mode  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-modes>`_
must be ```Default`` to allow multiple processes to use a single GPU.
If you use device oversubscription with NVIDIA GPUs the kernel executed from different processes will be serialized by the driver,
this is mostly describing the performance of PIConGPU because the device is under utilized.

Multi-Process Service (MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use `NVIDIA MPS <https://docs.nvidia.com/deploy/mps/index.html>`_ and split one device into ``4`` you need to use
``--numRanksPerDevice 4`` for PIConGPU even if MPS is providing you with ``4`` virtual gpus.
MPS can be used to workaround the kernel serialization when using multiple processes per GPU.

CPU
---

If you :ref:`compiled PIConGPU with a CPU accelerator <usage-basics-configure>` e.g. `omp2b`, `serial`, `tbb`, or `threads`
device oversubscribing will have no effect.
For CPU accelerators PIConGPU is not using a pre allocated device memory heap therefore you can freely choose the number
of MPI ranks per CPU.
