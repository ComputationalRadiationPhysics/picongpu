Introduction
============

The *alpaka* library defines and implements an abstract interface for the *hierarchical redundant parallelism* model.
This model exploits task- and data-parallelism as well as memory hierarchies at all levels of current multi-core architectures.
This allows to achieve portability of performant codes across various types of accelerators by ignoring specific unsupported levels and utilizing only the ones supported on a specific accelerator.
All hardware types (multi- and many-core CPUs, GPUs and other accelerators) are treated and can be programmed in the same way.
The *alpaka* library provides back-ends for *CUDA*, *OpenMP*, *Boost.Fiber* and other methods.
The policy-based C++ template interface provided allows for straightforward user-defined extension of the library to support other accelerators.

The library name *alpaka* is an acronym standing for **A**\ bstraction **L**\ ibrary for **Pa**\ rallel **K**\ ernel **A**\ cceleration.


Motivation
----------

What scales well on current hardware does not necessarily scale well on future architectures.
The hardware landscape is always changing.
In the past the big clusters have been CPU only.
Today we see a change to accelerator supported computing.
For example, GPUs, Intel Xeon Phis or other special purpose extension cards are extensively used.
It is unpredictable what the next big step will be and how the Exaflop hardware will look like.
It is not clear that GPUs will always be the best platform.
Nevertheless, the underlying physical algorithms as well as the need for heterogeneous architectures will not change.

Current highly parallel GPUs are optimized for throughput and hide latency and data dependencies by always keeping a ready pool of work.
This allows to sustain the performance at a high percent of peak.
CPUs in turn are designed to optimize the execution time of a single thread.
Features like branch prediction, speculative execution, register renaming and many more *[...] would cost far too much energy to be replicated for thousands of parallel GPU threads but [...] are entirely appropriate for CPUs.* (`State-of-the-art in Heterogeneous Computing`_)
Even more specialized architectures will appear and find their way into HPC

  *The essence of the heterogeneous computing model is that one size does not fit all. Parallel and serial segments of the workload execute on the best-suited processor delivering faster overall performance, greater efficiency, and lower energy and cost per unit of computation.* (`State-of-the-art in Heterogeneous Computing`_)

New hardware will not only allow to execute faster or calculate more but will furthermore enable the usage of new algorithms for more precise simulations.
For example, some tasks may require random searches for only a few values in a lookup table of up to hundreds of gigabytes.
This would perfectly fit to a CPUs, while the rest of the simulation would still be running on the GPUs.
With new hardware bringing those two worlds closer together, exploiting the heterogeneous hardware with heterogenous algorithms will likely be the way to go in the future.
Being able to express both of those parallel tasks in the same way would greatly enhance the productivity of the programmer and the clarity of the code.

Porting a complicated simulation code from *CUDA* to x86 and possibly to other hardware architectures is a non-trivial task.
A lot of developer time could be saved if this task would not have to be done repeatedly for every new hardware, but rather only once.
Therefore, *alpaka* tries to solve the problems in porting highly scalable simulation codes on various multi-core architectures.

.. _State-of-the-art in Heterogeneous Computing: https://dx.doi.org/10.1155/2010/540159

Problems in Porting Performant HPC Codes
----------------------------------------

Porting a highly performant code to a new architecture is a non-trivial task that poses many problems.
Often it is a requirement to keep the simulation operative on the previous platform as well.
This means that multiple hardware platforms have to be supported simultaneously.
A great number of projects take the route that seems easiest at first and simply duplicate all the parallel algorithms and port them to the new back-end.
All the specific API functions that have been used, have to be supplemented by the new pendants, possibly guarded by preprocessor macros to switch between the old and the new version.
A switch of the back-end used in a simulation, for example, from *OpenMP* to *CUDA* often requires a near rewrite.
Each newly supported platform would have to duplicate the API specific kernel and invocation code lines.

The following paragraphs will summarize problems that arise when performant HPC codes have to be ported:

Sustainability
~~~~~~~~~~~~~~

Because the underlying HPC hardware is constantly changing, every new generation will require an adaption of the simulation.
Even to deliver the performance reached on previous architectures is a tough task for programmers.
Furthermore, nobody can guarantee the lifespan of the parallelization technique used.
*OpenMP*, *CUDA*, *OpenACC* and all the other possibilities could be discontinued or get deprecated for any reason at any time.
Therefore, an abstract interface is required that hides the particular back-end and allows to port the interface implementation and not the application using the interface itself.

Heterogeneity
~~~~~~~~~~~~~

Some parts of a simulation perfectly map to current GPUs while other parts are better computed on CPUs or other accelerators.
Furthermore, by letting one part of the heterogeneous cluster hardware idle, a lot of computing power is wasted.
It is essential, especially for future architectures, that those resources are utilized to reach the peak performance of the systems.
This heterogeneous work division not only depends on the architecture but also on the number of available hardware resources, the workload and many other factors.
Therefore, to reach good scaling across a multitude of systems, it is necessary to be able to dynamically decide where to execute which part of the simulation either at make-time, compile-time or at run-time.
Currently this requires to duplicate the kernels and write specific implementations per back-end.
Many projects only allow to switch the back-end of the whole simulation at once or possibly even per kernel at make-time.
This will not be enough on future architectures where the ability to mix the back-ends is required to optimally utilize different cluster architectures or to dynamically load balance tasks across a diverse set of (possibly failing) accelerator devices.
Therefore, an abstract interface unifying the abilities of all the back-ends is required to let the application express parallelism of the different back-ends in a unified algorithm that can then be mapped to the device currently in use.

Maintainability
~~~~~~~~~~~~~~~

Looking at the software engineering aspects, duplication is a bad solution because this leads to maintainability issues.
In many projects such copies result in a large growth in the number of lines of code while only minimal new functionality is implemented.
Most of the new code only executes things that have already been implemented for the initial platform.
Developers having to change one of the algorithms additionally have to change all duplicates for all other back-ends.
Depending on the similarity of the implementations, this can result in a doubling / multiplication of developer efforts in the worst-case scenario.
Especially for open-source projects that rely on contributions from the community this raises the hurdle for new developers because they have to know not only one, but multiple different parallelization libraries.
In the end good maintainability is what keeps a software project alive and what ensures a steady development progress.
Therefore, an interface hiding the differences between all the back-ends is required to let the application express parallelism in a unified algorithm.

Testability
~~~~~~~~~~~

Code duplication, being the easiest way to port a simulation, exacerbates testing.
Each new kernel has to be tested separately because different bugs could have been introduced into the distinct implementations.
If the versions can be mixed, it is even harder because all combinations have to be tested.
Often the tests (continuous integration tests, unit tests, etc.) have to run on a special testing hardware or on the production systems due to the reliance on the availability of special accelerators.
For example, *CUDA* compile tests are possible without appropriate hardware but it is not feasible to execute even simple runtime tests due to the missing CPU emulation support.
An interface allowing to switch between acceleration back-ends, which are tested for compatibility among each other, enables easy testing on development and test systems.

Optimizability
~~~~~~~~~~~~~~

Even if the simulation code has encapsulated the APIs used, the optimal way to write performant algorithms often differs between distinct parallelization frameworks.
It is necessary to allow the user to fine-tune the algorithm to run optimally on each different accelerator device by compile time specialization or policy based abstractions without the need to duplicate the kernel.
Within the kernel there has to be knowledge about the underlying platform to adaptively use data structures that map optimally onto the current architecture.
To ease this optimization work, libraries with data structures, communication patterns and other things hiding the differences between back-ends have to be implemented.
This would allow to optimize the interface implementation and not the simulation itself.

In summary, it can be stated that all the portability problems of current HPC codes could be solved by introducing an abstract interface that hides the particular back-end implementations and unifies the way to access the parallelism available on modern many-core architectures.


Similar Projects
----------------

There are multiple other libraries targeting the (portable) parallel task execution within nodes.
Some of them require language extensions, others pretend to achieve full performance portability across a multitude of devices.
But none of these libraries can provide full control over the (possibly diverse) underlying hardware while being only minimal invasive.
There is always a productivity-performance trade-off.

Furthermore, many of the libraries do not satisfy the requirement for full single-source C++ support.
This is essential because many simulation codes heavily rely on template meta-programming for method specialization and compile time optimizations.


CUDA - Compute Unified Device Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*CUDA* is a parallel computing platform and programming model developed by *NVIDIA*.
It is used in science and research as well as in consumer software to compute highly parallel workloads on GPUs starting from image and video editing up to simulations on high-performance computers.
Such usage of graphics processing units not only for computer graphics, but also for tasks that have traditionally been handled by the CPU is called GPGPU (general-purpose computing on graphics processing units).
A disadvantage of *CUDA* is that its application is bound to the usage of *NVIDIA* GPUs.
Currently no other vendors provide accelerators that support *CUDA*.
Additionally there is no supported free emulator allowing to execute *CUDA* code on CPUs.

The *CUDA* API is a higher level part of the programming model which allows to access and execute code on GPUs from multiple host languages including C++.
The *CUDA* C/C++ language on the other hand is a mid level construct based on standard C++ with some extensions for accelerator programming and limitations in the supported constructs.
For example, throwing and catching exceptions as well as run-time type information (RTTI) are not supported.
*CUDA* C/C++ is compiled to a low level virtual instruction set called PTX (Parallel Thread Execution).
The PTX code is later compiled to assembler code by the GPU driver.

*NVIDIA* provides an extended C++ compiler based on the LLVM clang compiler called nvcc that allows to mix host C++ code using the *CUDA* API with *CUDA* C/C++.
The host part of the C++ code is compiled by the respective host system compiler (gcc, icc, clang, MSVC) while the GPU device code is separately compiled to PTX.
After the compilation steps both binaries are linked together to form the final assembly.

*CUDA* defines a heterogeneous programming model where tasks are offloaded from the host CPU to the device GPU.
Functions that should be offloaded to the GPU are called kernels.
As can be seen in the figure below a grid of such kernels is executed in parallel by multiple threads organized in blocks.
Threads within a block can synchronize, while blocks are executed independently and possibly in sequential order depending on the underlying hardware.

.. image:: https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png

The global device memory is the slowest but largest memory accessible by all threads.
It can be accessed from host code via methods provided by the *CUDA* API.
Global memory is persistent across kernel invocations.
Threads within a block can communicate through a fast but small shared memory.
Each thread has a set of very low latency registers similar to CPU threads.
Additionally there are special purpose memory sections for constant and texture data.

The *CUDA* C/C++ language gives full control over memory, caches and the execution of kernels.


`PGI CUDA-X86 <https://www.pgroup.com/resources/cuda-x86.htm>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a compiler technology that allows to generate x86-64 binary code from *CUDA* C/C++ applications using the *CUDA Runtime API* but does not support the *CUDA Driver API*.
At run-time *CUDA* C programs compiled for x86 execute each *CUDA* thread block using a single host core, eliminating synchronization where possible.
Multiple kernel threads are combined to be executed together via the CPUs SIMD (Single Instruction Multiple Data) capabilities for vectorized execution.
The *PGI Unified Binary technology* allows to create a single binary that uses *NVIDIA* GPUs when available, or runs on multi-core CPUs else.
The compiler is not always up-to-date with the latest *CUDA* versions and is not available for free.
Furthermore, the compiler seems not to be developed actively since *NVIDIA* acquired *PGI* in 2013.
Since 2012 no news were published and nothing could be found in the yearly release notes of the *PGI* compiler suite.


`GPU Ocelot <https://gpuocelot.gatech.edu/>`_ (`github <https://github.com/gtcasl/gpuocelot>`_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is an open-source dynamic JIT compilation framework.
It allows to execute native *CUDA* binaries by dynamically translating the *NVIDIA PTX* virtual instruction set architecture to other instruction sets.
It supports *NVIDIA* and *AMD* GPUs as well as multicore CPUs via a PTX to LLVM (Low Level Virtual Machine) translator.
The project is not in active development anymore.
It only supports PTX up to version 3.1 (current version is 5.0).


`OpenMP <https://www.openmp.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is an open specification for vendor agnostic shared memory parallelization.
By adding annotations (pragmas in C/C++) to loops or regions, it allows to easily parallelize existing sequential C/C++/Fortran code in an incremental manner.
Due to the nature of pragmas, these hints are ignored if the compiler does not support them or thinks they are inappropriate.
This allows those programs to be compiled as sequential or parallel versions by only changing a compiler flag.
In C/C++ the syntax for *OpenMP* directives is ``#pragma omp`` followed by multiple clauses.
For example, with the directive ``#pragma omp parallel for``, the compiler will automatically distribute the iterations of the directly following loop across the available cores.
*OpenMP* 4.0 introduced support for offloading computations to accelerator devices, substantially improved the task support and extended the SIMD capabilities.
By embedding code within a ``#pragma omp target`` block, the contained code will be executed on the selected device.
*OpenMP* 4.0 is missing the ability for unstructured data movement and only implements structured data movement from and to devices.
The compiler directive ``#pragma omp target data map(...) ...`` at the begin of a code block will define which data is copied to, copied back from and is created on the device.
At the end of the code block the memory is copied back or gets deleted.
There is no way to allocate device memory that is persistent between kernel calls in different methods because it is not possible to create a device data region spanning both functions in the general case.
*OpenMP* 4.1, expected for the end of 2015, is likely to introduce ``#pragma omp target enter data``, ``#pragma omp target exit data`` and other unstructured data movement directives that allow to pass and obtain pointers of already resident memory to and from offloaded kernels.
Currently *OpenMP* does not provide a way to control the hierarchical memory because its main assumption is a shared memory for all threads.
Therefore, the block shared memory on *CUDA* devices can not be explicitly utilized.


`OpenACC <https://www.openacc.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a pragma based programming standard for heterogeneous computing.
It is very similar to *OpenMP* and provides annotations for parallel execution and data movement as well as run-time functions for accelerator and device management.
In contrast to *OpenMP* it allows limited access to *CUDA* block shared memory.
Current compiler implementations support *NVIDA*, *AMD* and *Intel* accelerators.
Only as of *OpenACC* 2.0 explicit memory management and tiling is supported.
*OpenACC* does not support dynamic allocation of memory (``new``, ``delete``) in kernel code.
It is aimed to be fully merged with *OpenMP* at some point, but for now *OpenMP* 4.0 only introduced some parts of it.


`OpenCL <https://www.khronos.org/opencl/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a programming framework for heterogeneous platforms.
It is fully hardware independent and can utilize CPUs and GPUs of nearly all vendors.
This is achieved by compiling the *OpenCL* kernel code (or the standardized *SPIR* intermediate representation) at run-time by the platform driver into the native instruction set.
Versions prior to 2.1 (released in March 2015) did only support a C-like kernel language.
Version 2.1 introduced a subset of C++14.
*OpenCL* does not support single-source programming (combining C++ host code and accelerator code in a single file).
This is a precondition for templated kernels which are required for policy based generic programming.
It is necessary to note that *NVIDIA* seems to neglect their *OpenCL* implementation.
Support for version 1.2 has just been added in April 2015 after only three and a half years after the publication of the standard.
*OpenCL* does not support dynamic allocation of memory (``new``, ``delete``) in kernel code.


`SYCL <https://www.khronos.org/sycl>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a cross-platform abstraction layer based on *OpenCL*.
The main advantage over *OpenCL* itself is that it allows to write single-source heterogeneous programs.
It enables the usage of a single C++ template function for host and device code.
As of now there is no usable free compiler implementation available that has good support for multiple accelerator devices.


`C++ AMP (Accelerated Massive Parallelism) <https://msdn.microsoft.com/en-us/library/hh265136.aspx>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is an open specification from *Microsoft* currently implemented on top of *DirectX 11*.
It is a language extension requiring compiler support that allows to annotate C++ code that can then be run on multiple accelerators.
*C++ AMP* requires the usage of the ``array`` data structure or the ``array_view`` wrapper responsible for copying data to and from the accelerator devices.
The ``parallel_for_each`` function is responsible for offloading the provided function object whose ``operator()`` has to be annotated with ``restrict(amp)``.
The threads can access shared memory and synchronize.
The range of supported accelerator devices, plaforms and compilers is currently very limited.


`KOKKOS <https://github.com/kokkos>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. seealso::
   * https://www.xsede.org/documents/271087/586927/Edwards-2013-XSCALE13-Kokkos.pdf
   * https://trilinos.org/oldsite/events/trilinos_user_group_2013/presentations/2013-11-TUG-Kokkos-Tutorial.pdf
   * https://on-demand.gputechconf.com/supercomputing/2013/presentation/SC3103\_Towards-Performance-Portable-Applications-Kokkos.pdf
   * https://dx.doi.org/10.3233/SPR-2012-0343

provides an abstract interface for portable, performant shared memory-programming.
It is a C++ library that offers ``parallel_for``, ``parallel_reduce`` and similar functions for describing the pattern of the parallel tasks.
The execution policy determines how the threads are executed.
For example, this influences the sizes of blocks of threads or if static or dynamic scheduling should be used.
The library abstracts the kernel as a function object that can not have any user defined parameters for its ``operator()``.
Inconveniently, arguments have to be stored in members of the function object coupling algorithm and data together.
*KOKKOS* provides both, abstractions for parallel execution of code and data management.
Multidimensional arrays with a neutral indexing and an architecture dependent layout are available, which can be used, for example, to abstract the underlying hardwares preferred memory access scheme that could be row-major, column-major or even blocked.


`Thrust <https://thrust.github.io/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a parallel algorithms library resembling the C++ Standard Template Library (STL).
It allows to select either the *CUDA*, *TBB* or *OpenMP* back-end at make-time.
Because it is based on generic ``host_vector`` and ``device_vector`` container objects, it is tightly coupling the data structure and the parallelization strategy.
There exist many similar libraries such as `ArrayFire <https://arrayfire.com/>`_ (*CUDA*, *OpenCL*, native C++), `VexCL <https://github.com/ddemidov/vexcl/>`_ (*OpenCL*, *CUDA*), `ViennaCL <http://viennacl.sourceforge.net/>`_ (*OpenCL*, *CUDA*, *OpenMP*) and `hemi <https://github.com/harrism/hemi/>`_ (*CUDA*, native C++).

.. seealso::
   * Phalanx
     See `here <https://www.mgarland.org/files/papers/phalanx-sc12-preprint.pdf>`_
     It is very similar to *alpaka* in the way it abstracts the accelerators.
     C++ Interface provides CUDA, OpenMP, and GASNet back-ends
   * Aura
   * Intel TBB
   * U\PC++


Distinction of the *alpaka* Library
-----------------------------------

In the section about the problems we saw that all portability problems of current HPC codes could be solved with an abstract interface unifying the underlying accelerator back-ends.
The previous section showed that there is currently no project available that could solve all of the problems highlighted.
The C++ interface library proposed to solve all those problems is called *alpaka*.
The subsequent enumeration will summarize the purpose of the library:

*alpaka* is ...
~~~~~~~~~~~~~~~

* an **abstract interface** describing parallel execution on multiple hierarchy levels. It allows to implement a mapping to various hardware architectures but **is no optimal mapping itself**.

* sustainably solving portability (50% on the way to reach full performance portability)

* solving the **heterogeneity** problem. An identical algorithm / kernel can be executed on heterogeneous parallel systems by selecting the target device.

* reducing the **maintainability** burden by not requiring to duplicate all the parts of the simulation that are directly facing the parallelization framework. Instead, it allows to provide a single version of the algorithm / kernel that can be used by all back-ends. All the accelerator dependent implementation details are hidden within the *alpaka* library.

* simplifying the **testability** by enabling **easy back-end switching**. No special hardware is required for testing the kernels. Even if the simulation itself will always use the *CUDA* back-end, the tests can completely run on a CPU. As long as the *alpaka* library is thoroughly tested for compatibility between the acceleration back-ends, the user simulation code is guaranteed to generate identical results (ignoring rounding errors / non-determinism) and is portable without any changes.

* **optimizable**. Everything in *alpaka* can be replaced by user code to optimize for special use-cases.

* **extensible**. Every concept described by the *alpaka* abstraction can be implemented by users. Therefore it is possible to non-intrusively define new devices, queues, buffer types or even whole accelerator back-ends.

* **data structure agnostic**. The user can use and define arbitrary data structures.

*alpaka* is not ...
~~~~~~~~~~~~~~~~~~~

* an automatically **optimal mapping** of algorithms / kernels to various acceleration platforms. Except in trivial examples an optimal execution always depends on suitable selected data structure. An adaptive selection of data structures is a separate topic that has to be implemented in a distinct library.

* automatically **optimizing concurrent data accesses**.

* **handling** or hiding differences in arithmetic operations. For example, due to **different rounding** or different implementations of floating point operations, results can differ slightly between accelerators.

* **guaranteeing any determinism** of results. Due to the freedom of the library to reorder or repartition the threads within the tasks it is not possible or even desired to preserve deterministic results. For example, the non-associativity of floating point operations give non-deterministic results within and across accelerators.

The *alpaka* library is aimed at parallelization within nodes of a cluster.
It does not compete with libraries for distribution of processes across nodes and communication among those.
For these purposes libraries like MPI (Message Passing Interface) or others should be used.
MPI is situated one layer higher and can be combined with *alpaka* to facilitate the hardware of a whole heterogeneous cluster.
The *alpaka* library can be used for parallelization within nodes, MPI for parallelization across nodes.


Comparison
----------

The following table summarizes which of the problems mentioned in section about the problems can be solved by current intra-node parallelization frameworks and the proof-of-concept *alpaka* abstraction library.


    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | Framework / API | Open-Source | Free | Single-Source C++ | Portability | Heterogenity | Maintainability | Testability | Optimizability | Data structure agnostic |
    +=================+=============+======+===================+=============+==============+=================+=============+================+=========================+
    | CUDA            | --          | X    | X                 | --          | --           | --              | --          | X              | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | PGI CUDA-x86    | --          | --   | X                 | X           | ~~           | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | GPU Ocelot      | X           | X    | X                 | X           | ~~           | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | OpenMP          | X           | X    | X                 | X           | X            | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | OpenACC         | X           | X    | X                 | X           | X            | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | OpenCL          | X           | X    | --                | X           | X            | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | SYCL            | X           | {X}  | X                 | X           | X            | X               | X           | {X}            | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | C++AMP          | X           | X    | X                 | {X}         | X            | X               | X           | --             | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | KOKKOS          | X           | X    | X                 | X           | X            | X               | X           | --             | ~~                      |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | Thrust          | X           | X    | X                 | X           | ~~           | X               | X           | --             | --                      |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+
    | ****alpaka****  | X           | X    | X                 | X           | X            | X               | X           | X              | X                       |
    +-----------------+-------------+------+-------------------+-------------+--------------+-----------------+-------------+----------------+-------------------------+


Properties of intra-node parallelization frameworks and their ability to solve the problems in porting performant HPC codes. X = yes/fully solved, ~~ = partially solved, -- = no / not solved
