/**
 * \file
 * Copyright 2014-2018 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>

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
// This example is hard-coded to use the sequential executor.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

    // Define the index domain
    using Dim = alpaka::dim::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using QueueAcc = alpaka::queue::QueueCpuSync;

    // Select a device
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

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
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Allocate 3 host memory buffers
    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostC(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));

    // Initialize the host input vectors A and B
    Data * const pBufHostA(alpaka::mem::view::getPtrNative(bufHostA));
    Data * const pBufHostB(alpaka::mem::view::getPtrNative(bufHostB));
    Data * const pBufHostC(alpaka::mem::view::getPtrNative(bufHostC));

    // C++11 random generator for uniformly distributed numbers in {1,..,42}
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
    using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    alpaka::mem::view::copy(queue, bufAccA, bufHostA, extent);
    alpaka::mem::view::copy(queue, bufAccB, bufHostB, extent);
    alpaka::mem::view::copy(queue, bufAccC, bufHostC, extent);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    // Create the executor task.
    auto const exec(alpaka::kernel::createTaskExec<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccA),
        alpaka::mem::view::getPtrNative(bufAccB),
        alpaka::mem::view::getPtrNative(bufAccC),
        numElements));

    // Enqueue the kernel executor
    alpaka::queue::enqueue(queue, exec);

    // Copy back the result
    alpaka::mem::view::copy(queue, bufHostC, bufAccC, extent);

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

#else
    return EXIT_SUCCESS;
#endif
}
