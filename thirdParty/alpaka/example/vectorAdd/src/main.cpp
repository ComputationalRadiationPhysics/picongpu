/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
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
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TSize const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            auto const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TSize i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

auto main()
-> int
{
    using Val = float;
    using Size = std::size_t;

    using Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<1u>, Size>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using StreamAcc = alpaka::stream::StreamCpuSync;

    using PltfHost = alpaka::pltf::PltfCpu;

    Size const numElements(123456);

    // Create the kernel function object.
    VectorAddKernel kernel;

    // Get the host device.
    auto const devHost(
        alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Select a device to execute on.
    auto const devAcc(
        alpaka::pltf::getDevByIdx<PltfAcc>(0));

    // Get a stream on this device.
    StreamAcc stream(devAcc);

    // The data extent.
    alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Size> const extent(
        numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, Size> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            static_cast<Size>(3u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "VectorAddKernelTester("
        << " numElements:" << numElements
        << ", accelerator: " << alpaka::acc::getAccName<Acc>()
        << ", kernel: " << typeid(kernel).name()
        << ", workDiv: " << workDiv
        << ")" << std::endl;

    // Allocate host memory buffers.
    auto memBufHostA(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));
    auto memBufHostB(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));
    auto memBufHostC(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));

    // Initialize the host input vectors
    for (Size i(0); i < numElements; ++i)
    {
        alpaka::mem::view::getPtrNative(memBufHostA)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        alpaka::mem::view::getPtrNative(memBufHostB)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
    }

    // Allocate the buffers on the accelerator.
    auto memBufAccA(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));
    auto memBufAccB(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));
    auto memBufAccC(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));

    // Copy Host -> Acc.
    alpaka::mem::view::copy(stream, memBufAccA, memBufHostA, extent);
    alpaka::mem::view::copy(stream, memBufAccB, memBufHostB, extent);

    // Create the executor task.
    auto const exec(alpaka::exec::create<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(memBufAccA),
        alpaka::mem::view::getPtrNative(memBufAccB),
        alpaka::mem::view::getPtrNative(memBufAccC),
        numElements));

    // Profile the kernel execution.
    alpaka::stream::enqueue(stream, exec);

    // Copy back the result.
    alpaka::mem::view::copy(stream, memBufHostC, memBufAccC, extent);

    bool resultCorrect(true);
    auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostC));
    for(Size i(0u);
        i < numElements;
        ++i)
    {
        auto const & val(pHostData[i]);
        auto const correctResult(alpaka::mem::view::getPtrNative(memBufHostA)[i]+alpaka::mem::view::getPtrNative(memBufHostB)[i]);
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#elif BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"  // "comparing floating point with == or != is unsafe"
#endif
        if(val != correctResult)
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#elif BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif
        {
            std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
    }

    return EXIT_SUCCESS;
}
