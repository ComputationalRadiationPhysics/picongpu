Introduction
============

The *alpaka* library defines and implements an abstract interface for the *hierarchical redundant parallelism* model.
This model exploits task- and data-parallelism as well as memory hierarchies at all levels of current multi-core architectures.
This allows to achieve performance portability across various types of accelerators by ignoring specific unsupported levels and utilizing only the ones supported on a specific accelerator.
All hardware types (CPUs, GPUs and other accelerators) are treated and can be programmed in the same way.
The *alpaka* library provides back-ends for *CUDA*, *OpenMP*, *HIP*, *SYCL* and other technologies.
The trait-based C++ template interface provided allows for straightforward user-defined extension of the library to support other accelerators.

The library name *alpaka* is an acronym standing for **A**\ bstraction **L**\ ibrary for **Pa**\ rallel **K**\ ernel **A**\ cceleration.

About alpaka
------------

alpaka is ...
~~~~~~~~~~~~~

Abstract
   It describes parallel execution on multiple hierarchy levels. It allows to implement a mapping to various hardware architectures but is no optimal mapping itself.

Sustainable
   *alpaka* decouples the application from the availability of different accelerator frameworks in different versions, such as OpenMP, CUDA, HIP, etc. (50% on the way to reach full performance portability).

Heterogeneous
   An identical algorithm / kernel can be executed on heterogeneous parallel systems by selecting the target device. This allows the best performance for each algorithm and/or a good utilization of the system without major code changes.

Maintainable
   *alpaka* allows to provide a single version of the algorithm / kernel that can be used by all back-ends. There is no need for "copy and paste" kernels with different API calls for different accelerators. All the accelerator dependent implementation details are hidden within the *alpaka* library.

Testable
   Due to the easy back-end switch, no special hardware is required for testing the kernels. Even if the simulation itself always uses the *CUDA* back-end, the tests can completely run on a CPU. As long as the *alpaka* library is thoroughly tested for compatibility between the acceleration back-ends, the user simulation code is guaranteed to generate identical results (ignoring rounding errors / non-determinism) and is portable without any changes.

Optimizable
   Everything in *alpaka* can be replaced by user code to optimize for special use-cases.

Extensible
   Every concept described by the *alpaka* abstraction can be implemented by users. Therefore it is possible to non-intrusively define new devices, queues, buffer types or even whole accelerator back-ends.

Data Structure Agnostic
   The user can use and define arbitrary data structures.

alpaka does not ...
~~~~~~~~~~~~~~~~~~~

Automatically provide an optimal mapping of kernels to various acceleration platforms
   Except in trivial examples an optimal execution always depends on suitable selected data structures. An adaptive selection of data structures is a separate topic that has to be implemented in a distinct library.

Automatically optimize concurrent data access
   *alpaka* does not provide feature to create optimized memory layouts.

Handle differences in arithmetic operations
   For example, due to **different rounding** or different implementations of floating point operations, results can differ slightly between accelerators.

Guarantee determinism of results
   Due to the freedom of the library to reorder or repartition the threads within the tasks it is not possible or even desired to preserve deterministic results. For example, the non-associativity of floating point operations give non-deterministic results within and across accelerators.

The *alpaka* library is aimed at parallelization on shared memory, i.e. within nodes of a cluster.
It does not compete with libraries for distribution of processes across nodes and communication among those.
For these purposes libraries like MPI (Message Passing Interface) or others should be used.
MPI is situated one layer higher and can be combined with *alpaka* to facilitate the hardware of a whole heterogeneous cluster.
The *alpaka* library can be used for parallelization within nodes, MPI for parallelization across nodes.
