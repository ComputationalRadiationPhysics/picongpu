.. highlight:: cpp
   :linenothreshold: 5

Rationale
=========

Interface Distinction
---------------------

The *alpaka* library is different from other similar libraries (especially *CUDA*) in that it refrains from using implicit or hidden state.
This and other interface design decisions will be explained int the following paragraphs.

No Current Device:
++++++++++++++++++

The *CUDA* runtime API for example supplies a current device for each user code kernel-thread.
Working with multiple devices requires to call ``cudaSetDevice`` to change the current device whenever an operation should be executed on a non-current device.
Even the functions for creating a queue (``cudaStreamCreate``) or an event (``cudaEventCreate``) use the current device without any way to create them on a non current device.
In the case of an event this dependency is not obvious, since at the same time queues can wait for events from multiple devices allowing cross-device synchronization without any additional work.
So conceptually an event could also have been implemented device independently.
This can lead to hard to track down bugs due to the non-explicit dependencies, especially in multi-threaded code using multiple devices.

No Default Device:
++++++++++++++++++

In contrast to the *CUDA* runtime API *alpaka* does not provide a device by default per kernel-thread.
Especially in combination with *OpenMP* parallelized host code this keeps users from surprises.
The following code snippet shows that it does not necessarily do what one would expect.

.. code-block::

   cudaSetDevice(1);

   #pragma omp parallel for
   for(int i = 0; i<10; ++i)
   {
       kernel<<<blocks,threads>>>(i);
   }

Depending on what the *CUDA* runtime API selects as default device for each of the *OpenMP* threads (due to each of them having its own current device), not all of the kernels will necessarily run on device one.

In the *alpaka* library all such dependencies are made explicit.
All functions depending on a device require it to be given as a parameter.
The *alpaka* *CUDA* back-end checks before forwarding the calls to the *CUDA* runtime API whether the current device matches the given one and changes it if required.
The *alpaka* *CUDA* back-end does not reset the current device to the one prior to the method invocation out of performance considerations.
This has to be considered when native *CUDA* code is combined with *alpaka* code.

No Default Queue:
+++++++++++++++++

*CUDA* allows to execute commands without specifying a queue.
The default queue that is used synchronizes implicitly with all other queues on the device.
If a command queue is issued to the default, all other asynchronous queues have to wait before executing any new commands, even when they have been enqueued much earlier.
This can introduce hard to track down performance issues.
As of *CUDA* 7.0 the default queue can be converted to a non synchronizing queue with a compiler option.
Because concurrency is crucial for performance and users should think about the dependencies between their commands from begin on, *alpaka* does not provide such a default queue.
All asynchronous operations (kernel launches, memory copies and memory sets) require a queue to be executed in.

No Implicit Built-in Variables and Functions:
---------------------------------------------

Within *CUDA* device functions (functions annotated with `__global__` or `__device__`) built-in functions (`__syncthreads`, `__threadfence`, `atomicAdd`, ... ) and variables (`gridDim`, `blockIdx`, `blockDim`, `threadIdx`, `warpSize`, ...) are provided.

It would have been possible to emulate those implicit definitions by forcing the kernel function object to inherit from a class providing these functions and members.
However functions outside the kernel function object would then pose a problem.
They do not have access to those functions and members, the function object has inherited.
To circumvent this, the functions and members would have to be public, the inheritance would have to be public and a reference to the currently executing function object would have to be passed as parameter to external functions.
This would have been too cumbersome and inconsistent.
Therefore access to the accelerator is given to the user kernel function object via one special input parameter representing the accelerator.
After that this accelerator object can simply be passed to other functions.
The built-in variables can be accessed by the user via query functions on this accelerator.

  * Abandoning all the implicit and default state makes it much easier for users of the library to reason about their code. *

No Language Extensions:
-----------------------

Unlike *CUDA*, the *alpaka* library does not extend the C++ language with any additional variable qualifiers (`__shared__`, `__constant__`, `__device__`) defining the memory space.
Instead of those qualifiers *alpaka* provides accelerator functions to allocate memory in different the different memory spaces.

No Dimensionality Restriction:
------------------------------

*CUDA* always uses three-dimensional indices and extents, even though the task may only be one or two dimensional.
*OpenCL* on the other hand allows grid and block dimensions in the range [1,3] but does not provide corresponding n-dimensional indices, but rather provides functions like `get_global_id` or `get_local_id`, which require the dimension in which the one-dimensional ID is to be queried as a parameter.
By itself this is no problem, but how can be assured that a two-dimensional kernel is called with grid and block extents of the correct dimensionality at compile time?
How can it be assured that a kernel which only uses `threadIdx.x` or equivalently calls `get_global_id(0)` will not get called with two dimensional grid and block extents?
Because the result in such a case is undefined, and most of the time not wanted by the kernel author, this should be easy to check and reject at compile-time.
In *alpaka* all accelerators are templatized on the dimensionality.
This allows a two-dimensional image filter to assert that it is only called with a two dimensional accelerator.
Thereby the algorithms can check for supported dimensionality of the accelerator at compile time instead of runtime.
Furthermore with the dimension being a template parameter, the CPU back-end implementations are able to use only the number of nested loops really necessary instead of the 6 loops (2 x 3 loops for grid blocks and block threads), which are mandatory to emulate the *CUDA* threaded blocking scheme.

*By hiding all the accelerator functionality inside of the accelerator object that is passed to the user kernel, the user of the *alpaka* library is not faced with any non-standard C++ extensions.
Nevertheless the *CUDA* back-end internally uses those language extensions.*

Integral Sizes of Arbitrary Type:
---------------------------------

The type of sizes such as extents, indices and related variables are depending on a template parameter of the accelerator and connected classes.
This allows the kernel to be executed with sizes of arbitrary ranges.
Thereby it is possible to force the accelerator back-ends to perform all internal index, extent and other integral size depending computations with a given precision.
This is especially useful on current *NVIDIA* GPUs.
Even though they support 64-bit integral operations, they are emulated with multiple 32-bit operations.
This can be a huge performance penalty when the sizes of buffers, offsets, indices and other integral variables holding sizes are known to be limited.

No Synchronous (Blocking) and Asynchronous (Non-Blocking) Function Versions:
----------------------------------------------------------------------------

*CUDA* provides two versions of many of the runtime functions, for example, `cudaMemcpyAsync` and `cudaMemcpy`.
The asynchronous version requires a queue while the synchronous version does not need a queue parameter.
The asynchronous version immediately returns control back to the caller while the task is enqueued into the given queue and executed later in parallel to the host code.
The synchronous version waits for the task to finish before the function call returns control to the caller.
Inconsistently, all kernels in a *CUDA* program can only be started either asynchronously by default or synchronously if `CUDA_LAUNCH_BLOCKING` is defined.
There is no way to specify this on a per kernel basis.
To switch a whole application from asynchronous to synchronous calls, for example for debugging reasons, it is necessary to change the names of all the runtime functions being called as well as their parameters.
In *alpaka* this is solved by always enqueuing all tasks into a queue and not defining a default queue.
Non-blocking queues as well as blocking queues are provided for all devices.
Changes to the synchronicity of multiple tasks can be made on a per queue basis by changing the queue type at the place of creation.
There is no need to change any line of calling code.

Memory Management
-----------------

Memory buffers can not only be identified by the pointer to their first byte.
The C++ `new` and `malloc`, the *CUDA* `cudaMalloc` as well as the *OpenCL* `clCreateBuffer` functions all return a plain pointer.
This is not enough when working with multiple accelerators and multiple devices.
To know where a specific pointer was allocated, additional information has to be stored to uniquely identify a memory buffer on a specific device.
Memory copies between multiple buffers additionally require the buffer extents and pitches to be known.
Many APIs, for example *CUDA*, require the user to store this information externally.
To unify the usage, *alpaka* stores all the necessary information in a memory buffer object.

Acceleratable Functions
-----------------------

Many parallelization libraries / frameworks do not fully support the separation of the parallelization strategy from the algorithm itself.
*OpenMP*, for example, fully mixes the per thread algorithm and the parallelization strategy.
This can be seen in the source listing showing a simple AXPY computation with OpenMP.

.. code-block::

   template<
       typename TIdx,
       typename TElem>
   void axpyOpenMP(
       TIdx const n,
       TElem const alpha,
       TElem const * const X,
       TElem * const Y)
   {
       #pragma omp parallel for
       for (i=0; i<n; i++)
       {
           Y[i] = alpha * X[i] + Y[i];
       }
   }

Only one line of the function body, line 13, is the algorithm itself, while all surrounding lines represent the parallelization strategy.

*CUDA*, *OpenCL* and other libraries allow, at least to some degree, to separate the algorithm from the parallelization strategy.
They define the concept of a kernel representing the algorithm itself which is then parallelized depending on the underlying hardware.
The AXPY *CUDA* kernel source code shown in figure consists only of the code of one single iteration.

.. code-block::

   template<
       typename TIdx,
       typename TElem>
   __global__ void axpyCUDA(
       TIdx const n,
       TElem const alpha,
       TElem const * const X,
       TElem * const Y)
   {
       TIdx const i(blockIdx.x*blockDim.x + threadIdx.x)
       if(i < n)
       {
           Y[i] = alpha * X[i] + Y[i];
       }
   }

On the other hand the *CUDA* implementation is bloated with code handling the inherent blocking scheme.
Even if the algorithm does not utilize blocking, as it is the case here, the algorithm writer has to calculate the global index of the current thread by hand (line 10).
Furthermore, to support vectors larger then the predefined maximum number of threads per block (1024 for current *CUDA* devices), multiple blocks have to be used.
When the number of blocks does not divide the number of vector elements, it has to be assured that the threads responsible for the vector elements behind the given length, do not access the memory to prevent a possible memory access error.

By using the kernel concept, the parallelization strategy, whether all elements are executed in sequential order, in parallel or blocked is not hard coded into the algorithm itself.
The possibly multidimensional nested loops do not have to be written by the user.
For example, six loops would be required to emulate the *CUDA* execution pattern with a grid of blocks consisting of threads.

Furthermore the kernel concept breaks the algorithm down to the per element level.
Recombining multiple kernel iterations to loop over lines, columns, blocks or any other structure is always possible by changing the calling code and does not require a change of the kernel.
In contrast, by using *OpenMP* this would not be possible.
Therefore the *alpaka* interface builds on the kernel concept, being the body of the corresponding standard for loop executed in each thread.

Execution Domain Specifications
-------------------------------

*CUDA* requires the user to annotate its functions with execution domain specifications.
Functions that can only be executed on the GPU have to be annotated with ``__device__``, functions that can be executed on the host and on the GPU have to be annotated with ``__host__ __device__`` and host only functions can optionally be annotated with ``__host__``.
The nvcc *CUDA* compiler uses these annotations to decide with which back-ends a function has to be compiled.
Depending on the compiler in use, *alpaka* defines the macros  ``ALPAKA_FN_HOST``, ``ALPAKA_FN_ACC`` and ``ALPAKA_FN_HOST_ACC`` with the identical meaning which can be used in the same positions.
When the *CUDA* compiler is used, they are defined to their *CUDA* equivalents, else they are empty.

Kernel Function
---------------

Requirements
++++++++++++

- User kernels should be implemented independent of the accelerator.
- A user kernel has to have access to accelerator methods (synchronization within blocks, index retrieval, ...).
- For usage with CUDA, the kernel methods have to be attributed with ``__device__ __host__``.
- The user kernel has to fulfill ``std::is_trivially_copyable`` because only such objects can be copied into CUDA device memory.
  A trivially copyable class is a class that
  #. Has no non-trivial copy constructors(this also requires no virtual functions or virtual bases)
  #. Has no non-trivial move constructors
  #. Has no non-trivial copy assignment operators
  #. Has no non-trivial move assignment operators
  #. Has a trivial destructor
- For the same reason all kernel parameters have to fulfill ``std::is_trivially_copyable``, too.

Implementation Variants
+++++++++++++++++++++++

There are two possible ways to tell the kernel about the accelerator type:

#. The kernel is templated on the accelerator type ...

   * (+) This allows users to specialize them for different accelerators. (Is this is really necessary or desired?)
   * (-) The kernel has to be a class template. This does not allow C++ lambdas to be used as kernels because they are no templates themselves (but only their ``operator()`` can be templated).
   * (-) This prevents the user from instantiating an accelerator independent kernel before executing it.
     Because the memory layout in inheritance hierarchies is undefined a simple copy of the user kernel or its members to its specialized type is not possible platform independently.
     This would require a copy from UserKernel<TDummyAcc> to UserKernel<TAcc> to be possible.
     The only way to allow this would be to require the user to implement a templated copy constructor for every kernel.
     This is not allowed for kernels that should be copyable to a CUDA device because std::is_trivially_copyable requires the kernel to have no non-trivial copy constructors.

   a) ... and inherits from the accelerator.

     * (-) The kernel itself has to inherit at least protected from the accelerator to allow the KernelExecutor to access the Accelerator.

     * (-) How do accelerator functions called from the kernel (and not within the kernel class itself) access the accelerator methods?

     Casting this to the accelerator type and giving it as parameter is too much to require from the user.
   b) ... and the ``operator()`` has a reference to the accelerator as parameter.

     * (+) This allows to use the accelerator in functions called from the kernel (and not within the kernel class itself) to access the accelerator methods in the same way the kernel entry point function can.
     * (-) This would require an additional object (the accelerator) in device memory taking up valuable CUDA registers (opposed to the inheritance solution). At least on CUDA all the accelerator functions could be inlined nevertheless.

#. The ``operator()`` is templated on the accelerator type and has a reference to the accelerator as parameter.

  * (+) The kernel can be an arbitrary function object with ``ALPAKA_FN_HOST_ACC`` attributes.
  * (+) This would allow to instantiate the accelerator independent kernel and set its members before execution.
  * (+/-) usable with polymorphic lambdas.
  * (-) The ``operator()`` could be overloaded on the accelerator type but there is no way to specialize the whole kernel class itself, so it always has the same members.
  * (-) This would require an additional object (the accelerator) in device memory taking up valuable CUDA registers (opposed to the inheritance solution). At least on CUDA all the accelerator functions could be inlined nevertheless.

Currently we implement version 2.


Implementation Notes
++++++++++++++++++++

Unlike *CUDA*, the *alpaka* library does not differentiate between the kernel function that represents the entry point and other functions that can be executed on the accelerator.
The entry point function that has to be annotated with ``__global__`` in *CUDA* is internal to the *alpaka* *CUDA* back-end and is not exposed to the user.
It directly calls into the user supplied kernel function object whose invocation operator is declared with ``ALPAKA_FN_ACC``, which equals ``__device__`` in *CUDA*.
In this respect there is no difference between the kernel entry point function and any other accelerator function in *alpaka*.

The ``operator()`` of the kernel function object has to be ``const``.
This is especially important for the *CUDA* back-end, as it could possibly use the constant memory of the GPU to store the function object.
The constant memory is a fast, cached, read-only memory that is beneficial when all threads uniformly read from the same address at the same time.
In this case it is as fast as a read from a register.


Access to Accelerator-Dependent Functionality
+++++++++++++++++++++++++++++++++++++++++++++

There are two possible ways to implement access to accelerator dependent functionality inside a kernel:

* Making the functions/templates members of the accelerator (maybe by inheritance) and calling them like ``acc.syncBlockThreads()`` or ``acc.template getIdx<Grid, Thread, Dim1>()``.
  This would require the user to know and understand when to use the template keyword inside dependent type  object function calls.
* The functions are only light wrappers around traits that can be specialized taking the accelerator as first value (it can not be the last value because of the potential use of variadic arguments).
  The resulting code would look like ``sync(acc)`` or ``getIdx<Grid, Thread, Dim1>(acc)``.
  Internally these wrappers would call trait templates that are specialized for the specific accelerator e.g. ``template<typename TAcc> Sync{...};``

The second version is easier to understand and usually shorter to use in user code.


Index and Work Division
-----------------------

*CUDA* requires the user to calculate the global index of the current thread within the grid by hand (already shown as ``axpyCUDA``).
On the contrary, *OpenCL* provides the methods ``get_global_size``, ``get_global_id``, ``get_local_size`` and ``get_local_id``.
Called with the required dimension, they return the corresponding local or global index or extent (size).
In *alpaka* this idea is extended to all dimensions.
To unify the method interface and to avoid confusion between the differing terms and meanings of the functions in *OpenCL* and *CUDA*, in *alpaka* these methods are template functions.


Block Shared Memory
-------------------

Static Block Shared Memory
++++++++++++++++++++++++++

The size of block shared memory that is allocated inside the kernel is required to be given as compile time constant.
This is due to CUDA not allowing to allocate block shared memory inside a kernel at runtime.

Dynamic Block Shared Memory
+++++++++++++++++++++++++++

The size of the external block shared memory is obtained from a trait that can be specialized for each kernel.
The trait is called with the current kernel invocation parameters and the block-element extent prior to each kernel execution.
Because the block shared memory size is only ever constant or dependent on the block-element extent or the parameters of the invocation this has multiple advantages:

* It forces the separation of the kernel invocation from the calculation of the required block shared memory size.
* It lets the user write this calculation once instead of multiple times spread across the code.
