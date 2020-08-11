Abstraction
===========

.. note::

   Objective of the abstraction is to separate the parallelization strategy from the algorithm itself.
   Algorithm code written by users should not depend on any parallelization library or specific strategy.
   This would allow to exchange the parallelization back-end without any changes to the algorithm itself.
   Besides allowing to test different parallelization strategies this also makes it possible to port algorithms to new, yet unsupported, platforms.

Parallelism and memory hierarchies at all levels need to be exploited in order to achieve performance portability across various types of accelerators.
Within this chapter an abstraction will be derivated that tries to provide a maximum of parallelism while simultaneously considering implementability and applicability in hardware.

Looking at the current HPC hardware landscape, we often see nodes with multiple sockets/processors extended by accelerators like GPUs or Intel Xeon Phi, each with their own processing units.
Within a CPU or a Intel Xeon Phi there are cores with hyper-threads, vector units and a large caching infrastructure.
Within a GPU there are many small cores and only few caches.
Each entity in the hierarchy has access to different memories.
For example, each socket / processor manages its RAM, while the cores additionally have non-explicit access to L3, L2 and L1 caches.
On a GPU there are global, constant, shared and other memory types which all can be accessed explicitly.
The interface has to abstract from these differences without sacrificing speed on any platform.

A process running on a multi-socket node is the largest entity within *alpaka*.
The abstraction is only about the task and data parallel execution on the process/node level and down.
It does not provide any primitives for inter-node communication.
However, such libraries can be combined with *alpaka*.

An application process always has a main thread and is by definition running on the host.
It can access the host memory and various accelerator devices.
Such accelerators can be GPUs, Intel Xeon Phis, the host itself or other devices.
Thus, the host not necessarily has to be different from the accelerator device used for the computations.
For instance, an Intel Xeon Phi simultaneously can be the host and the accelerator device.

The *alpaka* library can be used to offload the parallel execution of task and data parallel work simultaneously onto different accelerator devices.

Task Parallelism
----------------

One of the basic building blocks of modern applications is task parallelism.
For example, the operating system scheduler, deciding which thread of which process gets how many processing time on which CPU core, enables task parallelism of applications.
It controls the execution of different tasks on different processing units.
Such task parallelism can be, for instance, the output of the progress in parallel to a download.
This can be implemented via two threads executing two different tasks.

The valid dependencies between tasks within an application can be defined as a DAG (directed acyclic graph) in all cases.
The tasks are represented by nodes and the dependencies by edges.
In this model, a task is ready to be executed if the number of incoming edges is zero.
After a task finished it's work, it is removed from the graph as well as all of it's outgoing edges,.
This reduces the number of incoming edges of subsequent tasks.

The problem with this model is the inherent overhead and the missing hardware and API support.
When it is directly implemented as a graph, at least all depending tasks have to be updated and checked if they are ready to be executed after a task finished.
Depending on the size of the graph and the number of edges this can be a huge overhead.

*OpenCL* allows to define a task graph in a somewhat different way.
Tasks can be enqueued into an out-of-order command queue combined with events that have to be finished before the newly enqueued task can be started.
Tasks in the command queue with unmet dependencies are skipped and subsequent ones are executed.
The ``CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`` property of a command queue is an optional feature only supported by few vendors.
Therefore, it can not be assumed to be available on all systems.

*CUDA* on the other hand does currently (version 7.5) not support such out-of-order queues in any way.
The user has to define dependencies explicitly through the order the tasks are enqueued into the queues (called queues in *CUDA*).
Within a queue, tasks are always executed in sequential order, while multiple queues are executed in parallel.
Queues can wait for events enqueued into other queues.

In both APIs, *OpenCL* and *CUDA*, a task graph can be emulated by creating one queue per task and enqueuing a unique event after each task, which can be used to wait for the preceding task.
However, this is not feasible due to the large queue and event creation costs as well as other overheads within this process.

Therefore, to be compatible with a wide range of APIs, the interface for task parallelism has to be constrained.
Instead of a general DAG, multiple queues of sequentially executed tasks will be used to describe task parallelism.
Events that can be enqueued into the queues enhance the basic task parallelism by enabling synchronization between different queues, devices or the host threads.

Data Parallelism
----------------

In contrast to task parallelism, data parallelism describes the execution of one and the same task on multiple, often related data elements.
For example, an image color space conversion is a textbook example of a data parallel task.
The same operation is executed independently on each pixel.
Other data parallel algorithms additionally introduce dependencies between threads in the input-, intermediate-, or output-data.
For example, the calculation of a brightness histogram has no input-data dependencies.
However, all pixel brightness values finally have to be merged into a single result.
Even these two simple examples show that it is necessary to think about the interaction of parallel entities to minimize the influence of data dependencies.

Furthermore, it is necessary to respect the principles of spatial and temporal locality.
Current hardware is built around these locality principles to reduce latency by using hierarchical memory as a trade-off between speed and hardware size.
Multiple levels of caches, from small and very fast ones to very large and slower ones exploit temporal locality by keeping recently referenced data as close to the actual processing units as possible.
Spatial locality in the main memory is also important for caches because they are usually divided into multiple lines that can only be exchanged one cache line at a time.
If one data element is loaded and cached, it is highly likely that nearby elements are also cached.
If the pixels of an image are stored row wise but are read out column wise, the spatial locality assumption of many CPUs is violated and the performance suffers.
GPUs on the other hand do not have a large caching hierarchy but allow explicit access to a fast memory shared across multiple cores.
Therefore, the best way to process individual data elements of a data parallel task is dependent on the data structure as well as the underlying hardware.

The main part of the *alpaka* abstraction is the way it abstracts data parallelism and allows the algorithm writer to take into account the hierarchy of processing units, their data parallel features and corresponding memory regions.
The abstraction developed is influenced and based on the groundbreaking *CUDA* and *OpenCL* abstractions of a multidimensional grid of threads with additional hierarchy levels in between.
Another level of parallelism is added to those abstractions to unify the data parallel capabilities of modern hardware architectures.
The explicit access to all hierarchy levels enables the user to write code that runs performant on all current platforms.
However, the abstraction does not try to automatically optimize memory accesses or data structures but gives the user full freedom to use data structures matching the underlying hardware preferences.

The individual levels are explained on the following pages:

.. toctree::
   :maxdepth: 1

   abstraction/thread
   abstraction/block
   abstraction/warp
   abstraction/element

Summary
-------

This abstraction is called *Redundant Hierarchical Parallelism*.
This term is inspired by the paper *The Future of Accelerator Programming: Abstraction, Performance or Can We Have Both?*
`PDF <http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/rocki_burtscher_sac14.pdf>`_
`DOI <https://dx.doi.org/10.1109/ICPADS.2013.76>`_
It investigates a similar *concept of copious parallel programming* reaching 80%-90% of the native performance while comparing CPU and GPU centric versions of an *OpenCL* n-body simulation with a general version utilizing parallelism on multiple hierarchy levels.

The *CUDA* or *OpenCL* abstractions themselves are very similar to the one designed in the previous sections and consists of all but the Element level.
However, as has been shown, all five abstraction hierarchy levels are necessary to fully utilize current architectures.
By emulating unsupported or ignoring redundant levels of parallelism, algorithms written with this abstraction can always be mapped optimally to all supported accelerators. The following table summarizes the characteristics of the proposed hierarchy levels.

    +-----------------+-----------------------+----------------+
    | Hierarchy Level | Parallelism           | Synchronizable |
    +-----------------+-----------------------+----------------+
    | ---             | ---                   | ---            |
    +-----------------+-----------------------+----------------+
    | grid            | sequential / parallel | -- / X         |
    +-----------------+-----------------------+----------------+
    | block           | parallel              | --             |
    +-----------------+-----------------------+----------------+
    | warp            | parallel              | X              |
    +-----------------+-----------------------+----------------+
    | thread          | parallel / lock-step  | X              |
    +-----------------+-----------------------+----------------+
    | element         | sequential            | --             |
    +-----------------+-----------------------+----------------+

Depending on the queue a task is enqueued into, grids will either run in sequential order within the same queue or in parallel in different queues.
They can be synchronized by using events.
Blocks can not be synchronized and therefore can use the whole spectrum of parallelism ranging from fully parallel up to fully sequential execution depending on the device.
Warps combine the execution of multiple threads in lock-step and can be synchronized implicitly by synchronizing the threads they contain.
Threads within a block are executed in parallel warps and each thread computes a number of data elements sequentially.

