/* Copyright 2019 Benjamin Worpitz, Erik Zenker
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <functional>

//-----------------------------------------------------------------------------
//! This functions says hi to the world and
//! can be encapsulated into a std::function
//! and used as a kernel function. It is
//! just another way to define alpaka kernels
//! and might be useful when it is necessary
//! to lift an existing function into a kernel
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
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
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
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8)));
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
    alpaka::exec<Acc>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC(Acc const& acc, size_t const nExclamationMarksAsArg) -> void {
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
