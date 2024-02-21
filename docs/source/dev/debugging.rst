.. _development-debugging:

Debugging
=========

.. sectionauthor:: Sergei Bastrakov

When investigating the reason of a crashing simulation, it is very helpful to reduce the setup as much as possible while the crash is still happening.
This includes moving to fewer (and ideally to 1) MPI processes, having minimal grid size (ideally 3 supercells), disabling unrelated plugins and simulation stages.
Such a reduction should be done incrementally while checking whether the issue is still observed.
Knowing when the issue disappears could hint to its source.
This process also significantly speeds up and helps to focus a following deeper look.

The following build options can assist the investigation:

* ``PIC_VERBOSE=<N>`` sets log detail level for PIConGPU, highest level is 127.
* ``PMACC_VERBOSE=<N>`` sets log detail level for pmacc, highest level is 127.
* ``PMACC_BLOCKING_KERNEL=ON`` makes each kernel invocation blocking, which helps to narrow a crash down to a particular kernel.
* ``CUPLA_STREAM_ASYNC_ENABLE=OFF`` disables asynchronous streams, also helps to narrow a crash down to a particular place.
* ``CMAKE_BUILD_TYPE=Debug`` compile in debug mode where all assertions are activated. Compiling in debug mode will slowdown your kernels!

These options can be passed when building, or manually modified via cmake.
An example build command is

.. code:: bash

   pic-build -c "-DPIC_VERBOSE=127 -DPMACC_VERBOSE=127 -DPMACC_BLOCKING_KERNEL=ON -DPMACC_USE_ASYNC_QUEUES=OFF"

When reporting a crash, it is helpful if you attached the output ``stdout`` and ``stderr`` of such a build.

For further debugging tips and use of tools please refer to `our Wiki <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/Debugging>`_.
