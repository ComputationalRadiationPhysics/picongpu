Cheatsheet
==========

.. only:: html

   Download pdf version :download:`here <../../cheatsheet/cheatsheet.pdf>`

General
-------

- Getting alpaka: https://github.com/alpaka-group/alpaka
- Issue tracker, questions, support: https://github.com/alpaka-group/alpaka/issues
- All alpaka names are in namespace alpaka and header file `alpaka/alpaka.hpp`
- This document assumes

  .. code-block:: c++

     #include <alpaka/alpaka.hpp>
     using namespace alpaka;

.. raw:: pdf

   Spacer 0,5

Accelerator, Platform and Device
--------------------------------

Define in-kernel thread indexing type
  .. code-block:: c++

    using Dim = DimInt<constant>;
    using Idx = IntegerType;

Define accelerator type (CUDA, OpenMP,etc.)
  .. code-block:: c++

    using Acc = AcceleratorType<Dim,Idx>;

  AcceleratorType:
     .. code-block:: c++

	AccGpuCudaRt,
	AccGpuHipRt,
	AccCpuSycl,
	AccFpgaSyclIntel,
	AccGpuSyclIntel,
	AccCpuOmp2Blocks,
	AccCpuOmp2Threads,
	AccCpuTbbBlocks,
	AccCpuThreads,
	AccCpuSerial


Create platform and select a device by index
   .. code-block:: c++

      auto const platform = Platform<Acc>{};
      auto const device = getDevByIdx(platform, index);

Queue and Events
----------------

Create a queue for a device
  .. code-block:: c++

    using Queue = Queue<Acc, Property>;
    auto queue = Queue{device};

  Property:
     .. code-block:: c++

	Blocking
	NonBlocking

Put a task for execution
  .. code-block:: c++

    enqueue(queue, task);

Wait for all operations in the queue
  .. code-block:: c++

    wait(queue);

Create an event
  .. code-block:: c++

     Event<Queue> event{device};

Put an event to the queue
  .. code-block:: c++

     enqueue(queue, event);

Check if the event is completed
  .. code-block:: c++

     isComplete(event);

Wait for the event (and all operations put to the same queue before it)
  .. code-block:: c++

     wait(event);

Memory
------

Memory allocation and transfers are symmetric for host and devices, both done via alpaka API

Create a CPU device for memory allocation on the host side
  .. code-block:: c++

     auto const platformHost = PlatformCpu{};
     auto const devHost = getDevByIdx(platformHost, 0);

Allocate a buffer in host memory
  .. code-block:: c++

     Vec<Dim, Idx> extent = value;
     using BufHost = Buf<DevHost, DataType, Dim, Idx>;
     BufHost bufHost = allocBuf<DataType, Idx>(devHost, extent);

(Optional, affects CPU – GPU memory copies) Prepare it for asynchronous memory copies
  .. code-block:: c++

     prepareForAsyncCopy(bufHost);

Create a view to host memory represented by a pointer
  .. code-block:: c++

     using Dim = alpaka::DimInt<1u>;
     Vec<Dim, Idx> extent = size;
     DataType* ptr = ...;
     auto hostView = createView(devHost, ptr, extent);

Create a view to host std::vector
   .. code-block:: c++

     auto vec = std::vector<DataType>(42u);
     auto hostView = createView(devHost, vec);

Create a view to host std::array
   .. code-block:: c++

     std::array<DataType, 2> array = {42u, 23};
     auto hostView = createView(devHost, array);

Get a raw pointer to a buffer or view initialization, etc.
  .. code-block:: c++

     DataType* raw = view::getPtrNative(bufHost);
     DataType* rawViewPtr = view::getPtrNative(hostView);

Allocate a buffer in device memory
  .. code-block:: c++

     auto bufDevice = allocBuf<DataType, Idx>(device, extent);

Enqueue a memory copy from host to device
  .. code-block:: c++

     memcpy(queue, bufDevice, bufHost, extent);

Enqueue a memory copy from device to host
  .. code-block:: c++

     memcpy(queue, bufHost, bufDevice, extent);

.. raw:: pdf

   PageBreak

Kernel Execution
----------------

Automatically select a valid kernel launch configuration
  .. code-block:: c++

     Vec<Dim, Idx> const globalThreadExtent = vectorValue;
     Vec<Dim, Idx> const elementsPerThread = vectorValue;

     auto autoWorkDiv = getValidWorkDiv<Acc>(
       device,
       globalThreadExtent, elementsPerThread,
       false,
       GridBlockExtentSubDivRestrictions::Unrestricted);

Manually set a kernel launch configuration
  .. code-block:: c++

     Vec<Dim, Idx> const blocksPerGrid = vectorValue;
     Vec<Dim, Idx> const threadsPerBlock = vectorValue;
     Vec<Dim, Idx> const elementsPerThread = vectorValue;

     using WorkDiv = WorkDivMembers<Dim, Idx>;
     auto manualWorkDiv = WorkDiv{blocksPerGrid,
                                  threadsPerBlock,
				  elementsPerThread};

Instantiate a kernel and create a task that will run it (does not launch it yet)
  .. code-block:: c++

     Kernel kernel{argumentsForConstructor};
     auto taskRunKernel = createTaskKernel<Acc>(workDiv, kernel, parameters);

acc parameter of the kernel is provided automatically, does not need to be specified here

Put the kernel for execution
  .. code-block:: c++

     enqueue(queue, taskRunKernel);

Kernel Implementation
---------------------

Define a kernel as a C++ functor
  .. code-block:: c++

     struct Kernel {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const & acc, parameters) const { ... }
     };

``ALPAKA_FN_ACC`` is required for kernels and functions called inside, ``acc`` is mandatory first parameter, its type is the template parameter

Access multi-dimensional indices and extents of blocks, threads, and elements
  .. code-block:: c++

     auto idx = getIdx<Origin, Unit>(acc);
     auto extent = getWorkDiv<Origin, Unit>(acc);
     // Origin: Grid, Block, Thread
     // Unit: Blocks, Threads, Elems

Access components of and destructuremulti-dimensional indices and extents
  .. code-block:: c++

     auto idxX = idx[0];
     auto [z, y, x] = extent3D;

Linearize multi-dimensional vectors
  .. code-block:: c++

     auto linearIdx = mapIdx<1u>(idx, extent);

.. raw:: pdf

   Spacer 0,8

Allocate static shared memory variable
  .. code-block:: c++

     Type& var = declareSharedVar<Type, __COUNTER__>(acc);       // scalar
     auto& arr = declareSharedVar<float[256], __COUNTER__>(acc); // array

Get dynamic shared memory pool, requires the kernel to specialize
  .. code-block:: c++

     trait::BlockSharedMemDynSizeBytes
       Type * dynamicSharedMemoryPool = getDynSharedMem<Type>(acc);

Synchronize threads of the same block
  .. code-block:: c++

     syncBlockThreads(acc);

Atomic operations
  .. code-block:: c++

     auto result = atomicOp<Operation>(acc, arguments);
     // Operation: AtomicAdd, AtomicSub, AtomicMin, AtomicMax, AtomicExch,
     //            AtomicInc, AtomicDec, AtomicAnd, AtomicOr, AtomicXor, AtomicCas
     // Also dedicated functions available, e.g.:
     auto old = atomicAdd(acc, ptr, 1);

Memory fences on block-, grid- or device level (guarantees LoadLoad and StoreStore ordering)
  .. code-block:: c++

     mem_fence(acc, memory_scope::Block{});
     mem_fence(acc, memory_scope::Grid{});
     mem_fence(acc, memory_scope::Device{});

Warp-level operations
  .. code-block:: c++

     uint64_t result = warp::ballot(acc, idx == 1 || idx == 4);
     assert( result == (1<<1) + (1<<4) );

     int32_t valFromSrcLane = warp::shfl(val, srcLane);

Math functions take acc as additional first argument
  .. code-block:: c++

     math::sin(acc, argument);

Similar for other math functions.

Generate random numbers
  .. code-block:: c++

     auto distribution = rand::distribution::createNormalReal<double>(acc);
     auto generator = rand::engine::createDefault(acc, seed, subsequence);
     auto number = distribution(generator);
