/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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

#include <random>
#include <iostream>
#include <typeinfo>

//#############################################################################
//! A vector addition kernel.
class VectorAddKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TIdx const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

auto main()
-> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the index domain
    using Dim = alpaka::dim::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators
    // that are defined in the alpaka::acc namespace e.g.:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuOmp4
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    // using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::acc::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::queue::Blocking;
    using QueueAcc = alpaka::queue::Queue<Acc, QueueProperty>;

    // Select a device
    auto const devAcc = alpaka::pltf::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    Idx const numElements(123456);
    Idx const elementsPerThread(3u);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    auto const devHost = alpaka::pltf::getDevByIdx<DevHost>(0u);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostC(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));

    // Initialize the host input vectors A and B
    Data * const pBufHostA(alpaka::mem::view::getPtrNative(bufHostA));
    Data * const pBufHostB(alpaka::mem::view::getPtrNative(bufHostB));
    Data * const pBufHostC(alpaka::mem::view::getPtrNative(bufHostC));

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{ rd() };
    std::uniform_int_distribution<Data> dist(1, 42);

    for (Idx i(0); i < numElements; ++i)
    {
        pBufHostA[i] = dist(eng);
        pBufHostB[i] = dist(eng);
        pBufHostC[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::mem::buf::Buf<Acc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    alpaka::mem::view::copy(queue, bufAccA, bufHostA, extent);
    alpaka::mem::view::copy(queue, bufAccB, bufHostB, extent);
    alpaka::mem::view::copy(queue, bufAccC, bufHostC, extent);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccA),
        alpaka::mem::view::getPtrNative(bufAccB),
        alpaka::mem::view::getPtrNative(bufAccC),
        numElements));

    // Enqueue the kernel execution task
    alpaka::queue::enqueue(queue, taskKernel);

    // Copy back the result
    alpaka::mem::view::copy(queue, bufHostC, bufAccC, extent);
    alpaka::wait::wait(queue);

    bool resultCorrect(true);
    for(Idx i(0u);
        i < numElements;
        ++i)
    {
        Data const & val(pBufHostC[i]);
        Data const correctResult(pBufHostA[i] + pBufHostB[i]);
        if(val != correctResult)
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
#endif
}
