/* Copyright 2023 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <functional>

//! This functions says hi to the world and
//! can be called from within a kernel function.
//! It might be useful when it is necessary
//! to lift an existing function into a device
//! function.
template<typename TAcc>
void ALPAKA_FN_ACC hiWorldFunction(TAcc const& acc, size_t const nExclamationMarks)
{
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

    Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

    printf(
        "[z:%u, y:%u, x:%u][linear:%u] Hi world from a function",
        static_cast<unsigned>(globalThreadIdx[0]),
        static_cast<unsigned>(globalThreadIdx[1]),
        static_cast<unsigned>(globalThreadIdx[2]),
        static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

    for(size_t i = 0; i < nExclamationMarks; ++i)
    {
        printf("!");
    }

    printf("\n");
}

auto main() -> int
{
// It requires support for extended lambdas when using nvcc as CUDA compiler.
// Requires sequential backend if CI is used
#if(!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__)))                                 \
    && (!defined(ALPAKA_CI) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED))

    // Define the index domain
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
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
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

    const size_t nExclamationMarks = 10;

    // Run "Hello World" kernel with a lambda function
    //
    // alpaka is able to execute lambda functions (anonymous functions).
    // alpaka forces the lambda function to accept
    // the utilized accelerator as first argument.
    // All following arguments can be provided after
    // the lambda function declaration or be captured.
    //
    // This example passes the number exclamation marks, that should
    // be written after we greet the world, to the
    // lambda function.
    //
    // To define a fully generic kernel lambda, the type of acc must be
    // auto. The Nvidia nvcc does not support generic lambdas, so the
    // type is set to Acc.
    alpaka::exec<Acc>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC(Acc const& acc, size_t const nExclamationMarksAsArg) -> void
        {
            auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
            auto linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

            printf(
                "[z:%u, y:%u, x:%u][linear:%u] Hello world from a lambda",
                static_cast<unsigned>(globalThreadIdx[0]),
                static_cast<unsigned>(globalThreadIdx[1]),
                static_cast<unsigned>(globalThreadIdx[2]),
                static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

            for(size_t i = 0; i < nExclamationMarksAsArg; ++i)
            {
                printf("!");
            }

            printf("\n");
        },
        nExclamationMarks);
    alpaka::wait(queue);

    return EXIT_SUCCESS;

#else
    return EXIT_SUCCESS;
#endif
}
