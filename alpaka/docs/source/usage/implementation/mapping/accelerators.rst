Accelerator Implementations
===========================
.. table::

    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | alpaka                                     | Serial                                        | std::thread                                                                   | Boost.Fiber                                                                  | OpenMP 2.0                                                                      | OpenMP 4.0                                                                                                                         | CUDA 9.0+                                        |
    +============================================+===============================================+===============================================================================+==============================================================================+=================================================================================+====================================================================================================================================+==================================================+
    | Devices                                    | Host Core                                     | Host Cores                                                                    | Host Core                                                                    | Host Cores                                                                      | Host Cores                                                                                                                         | NVIDIA GPUs                                      |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | Lib/API                                    | n/a                                           | std::thread                                                                   | boost::fibers::fiber                                                         | OpenMP 2.0                                                                      | OpenMP 4.0                                                                                                                         | CUDA 9.0+                                        |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | Kernel execution                           | n/a                                           | std::thread(kernel)                                                           | boost::fibers::fiber(kernel)                                                 | ompsetdynamic(0), #pragma omp parallel numthreads(iNumKernelsInBlock)           | #pragma omp target, #pragma omp teams numteams(...) threadlimit(...), #pragma omp distribute, #pragma omp parallel numthreads(...) | cudaConfigureCall, cudaSetupArgument, cudaLaunch |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | Execution strategy grid-blocks             | sequential                                    | sequential                                                                    | sequential                                                                   | sequential                                                                      | undefined                                                                                                                          | undefined                                        |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | Execution strategy block-kernels           | sequential                                    | preemptive multitasking                                                       | cooperative multithreading                                                   | preemptive multitasking                                                         | preemptive multitasking                                                                                                            | lock-step within warps                           |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | getIdx                                     | n/a                                           | block-kernel: mapping of std::thisthread::getid() grid-block: member variable | block-kernel: mapping of std::thisfiber::getid() grid-block: member variable | block-kernel: ompgetthreadnum() to 3D index mapping grid-block: member variable | block-kernel: ompgetthreadnum() to 3D index mapping grid-block: member variable                                                    | threadIdx, blockIdx                              |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | getExtent                                  | member variables                              | member variables                                                              | member variables                                                             | member variables                                                                | member variables                                                                                                                   | gridDim, blockDim                                |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | getBlockSharedExternMem                    | allocated in memory prior to kernel execution | allocated in memory prior to kernel execution                                 | allocated in memory prior to kernel execution                                | allocated in memory prior to kernel execution                                   | allocated in memory prior to kernel execution                                                                                      | _shared__                                        |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | allocBlockSharedMem                        | master thread allocates                       | syncBlockKernels -> master thread allocates -> syncBlockKernels               | syncBlockKernels -> master thread allocates -> syncBlockKernels              | syncBlockKernels -> master thread allocates -> syncBlockKernels                 | syncBlockKernels -> master thread allocates -> syncBlockKernels                                                                    | _shared__                                        |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | syncBlockKernels                           | n/a                                           | barrier                                                                       | barrier                                                                      | #pragma omp barrier                                                             | #pragma omp barrier                                                                                                                | _syncthreads                                     |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | atomicOp                                   | n/a                                           | std::lockguard< std::mutex >                                                  | n/a                                                                          | #pragma omp critical                                                            | #pragma omp critical                                                                                                               | atomicXXX                                        |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
    | ALPAKAFNHOSTACC, ALPAKAFNACC, ALPAKAFNHOST | inline                                        | inline                                                                        | inline                                                                       | inline                                                                          | inline                                                                                                                             | _device__, host, forceinline                     |
    +--------------------------------------------+-----------------------------------------------+-------------------------------------------------------------------------------+------------------------------------------------------------------------------+---------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+



Serial
------

The serial accelerator only allows blocks with exactly one thread.
Therefore it does not implement real synchronization or atomic primitives.

Threads
-------

Execution
+++++++++

To prevent recreation of the threads between execution of different blocks in the grid, the threads are stored inside a thread pool.
This thread pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage and lots of idling kernel-threads when there are multiple KernelExecutors around.
Because the default policy of the threads in the pool is to yield instead of waiting, this would also slow down the system immensely.

Fibers
------

Execution
+++++++++

To prevent recreation of the fibers between execution of different blocks in the grid, the fibers are stored inside a fibers pool.
This fiber pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage when there are multiple KernelExecutors around.

OpenMP
------

Execution
+++++++++

Parallel execution of the kernels in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
So we have to spawn one real thread per kernel in a block.
``omp for`` is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
Therefore we use ``omp parallel`` with the specified number of threads in a block.
Another reason for not using ``omp for`` like ``#pragma omp parallel for collapse(3) num_threads(blockDim.x*blockDim.y*blockDim.z)`` is that ``#pragma omp barrier`` used for intra block synchronization is not allowed inside ``omp for`` blocks.

Because OpenMP is designed for a 1:1 abstraction of hardware to software threads, the block size is restricted by the number of OpenMP threads allowed by the runtime.
This could be as little as 2 or 4 kernels but on a system with 4 cores and hyper-threading OpenMP can also allow 64 threads.

Index
+++++

OpenMP only provides a linear thread index. This index is converted to a 3 dimensional index at runtime.

Atomic
++++++

We can not use '#pragma omp atomic' because braces or calling other functions directly after ``#pragma omp atomic`` are not allowed.
Because we are implementing the CUDA atomic operations which return the old value, this requires ``#pragma omp critical`` to be used.
``omp_set_lock`` is an alternative but is usually slower.

CUDA
----

Nearly all CUDA functionality can be directly mapped to alpaka function calls.
A major difference is that CUDA requires the block and grid sizes to be given in (x, y, z) order.
alpaka uses the mathematical C/C++ array indexing scheme [z][y][x].
Dimension 0 in this case is z, dimensions 2 is x.

Furthermore alpaka does not require the indices and extents to be 3-dimensional.
The accelerators are templatized on and support arbitrary dimensionality.
NOTE: Currently the CUDA implementation is restricted to a maximum of 3 dimensions!

NOTE: The CUDA-accelerator back-end can change the current CUDA device and will NOT set the device back to the one prior to the invocation of the alpaka function!
