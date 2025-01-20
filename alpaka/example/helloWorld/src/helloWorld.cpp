/* Copyright 2023 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

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

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    //
    // Depending on your type of problem, you have to define
    // the dimensionality as well as the type used for indices.
    // For small index domains 16 or 32 bit indices may be enough
    // and may be faster to calculate depending on the accelerator.
    using Dim = alpaka::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
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
    auto const elementsPerGrid = Vec{4, 2, 4};

    // Instantiate the kernel function object
    //
    // Kernels can be everything that is trivially copyable, has a
    // callable operator() and takes the accelerator as first
    // argument. So a kernel can be a class or struct, a lambda, etc.
    HelloWorldKernel helloWorldKernel;

    alpaka::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, helloWorldKernel);

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
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
