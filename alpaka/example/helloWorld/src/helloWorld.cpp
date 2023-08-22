/* Copyright 2023 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

//! Hello World Kernel
//!
//! Prints "[x, y, z][gtid] Hello World" where tid is the global thread number.
struct HelloWorldKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

        // In the most cases the parallel work distibution depends
        // on the current index of a thread and how many threads
        // exist overall. These information can be obtained by
        // getIdx() and getWorkDiv(). In this example these
        // values are obtained for a global scope.
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Map the three dimensional thread index into a
        // one dimensional thread index space. We call it
        // linearize the thread index.
        Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        // Each thread prints a hello world to the terminal
        // together with the global index of the thread in
        // each dimension and the linearized global index.
        // Mind, that alpaka uses the mathematical index
        // order [z][y][x] where the last index is the fast one.
        printf(
            "[z:%u, y:%u, x:%u][linear:%u] Hello World\n",
            static_cast<unsigned>(globalThreadIdx[0u]),
            static_cast<unsigned>(globalThreadIdx[1u]),
            static_cast<unsigned>(globalThreadIdx[2u]),
            static_cast<unsigned>(linearizedGlobalThreadIdx[0u]));
    }
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Define the index domain
    //
    // Depending on your type of problem, you have to define
    // the dimensionality as well as the type used for indices.
    // For small index domains 16 or 32 bit indices may be enough
    // and may be faster to calculate depending on the accelerator.
    using Dim = alpaka::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // Each accelerator has strengths and weaknesses. Therefore,
    // they need to be chosen carefully depending on the actual
    // use case. Furthermore, some accelerators only support a
    // particular workdiv, but workdiv can also be generated
    // automatically.

    // By exchanging the Acc and Queue types you can select where to execute the kernel.
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    //
    // The accelerator only defines how something should be
    // parallelized, but a device is the real entity which will
    // run the parallel program. The device can be chosen
    // by id (0 to the number of devices minus 1) or you
    // can also retrieve all devices in a vector (getDevs()).
    // In this example the first devices is chosen.
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    //
    // A queue can be interpreted as the work queue
    // of a particular device. Queues are filled with
    // tasks and alpaka takes care that these
    // tasks will be executed. Queues are provided in
    // non-blocking and blocking variants.
    // The example queue is a blocking queue to a cpu device,
    // but it also exists as non-blocking queue for this
    // device (QueueCpuNonBlocking).
    Queue queue(devAcc);

    // Define the work division
    //
    // A kernel is executed for each element of a
    // n-dimensional grid distinguished by the element indices.
    // The work division defines the number of kernel instantiations as
    // well as the type of parallelism used by the kernel execution task.
    // Different accelerators have different requirements on the work
    // division. For example, the sequential accelerator can not
    // provide any thread level parallelism (synchronizable as well as non synchronizable),
    // whereas the CUDA accelerator can spawn hundreds of synchronizing
    // and non synchronizing threads at the same time.
    //
    // The workdiv is divided in three levels of parallelization:
    // - grid-blocks:      The number of blocks in the grid (parallel, not synchronizable)
    // - block-threads:    The number of threads per block (parallel, synchronizable).
    //                     Each thread executes one kernel invocation.
    // - thread-elements:  The number of elements per thread (sequential, not synchronizable).
    //                     Each kernel has to execute its elements sequentially.
    //
    // - Grid     : consists of blocks
    // - Block    : consists of threads
    // - Elements : consists of elements
    //
    // Threads in the same grid can access the same global memory,
    // while threads in the same block can access the same shared
    // memory. Elements are supposed to be used for vectorization.
    // Thus, a thread can process data element size wise with its
    // vector processing unit.
    using Vec = alpaka::Vec<Dim, Idx>;
    auto const elementsPerThread = Vec::all(static_cast<Idx>(1));
    auto const threadsPerGrid = Vec{4, 2, 4};
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Instantiate the kernel function object
    //
    // Kernels can be everything that is trivially copyable, has a
    // callable operator() and takes the accelerator as first
    // argument. So a kernel can be a class or struct, a lambda, etc.
    HelloWorldKernel helloWorldKernel;

    // Run the kernel
    //
    // To execute the kernel, you have to provide the
    // work division as well as the additional kernel function
    // parameters.
    // The kernel execution task is enqueued into an accelerator queue.
    // The queue can be blocking or non-blocking
    // depending on the chosen queue type (see type definitions above).
    // Here it is synchronous which means that the kernel is directly executed.
    alpaka::exec<Acc>(
        queue,
        workDiv,
        helloWorldKernel
        /* put kernel arguments here */);
    alpaka::wait(queue);

    return EXIT_SUCCESS;
#endif
}
