/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <iostream>
#include <typeinfo>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>

//#############################################################################
//! A vector addition kernel.
class AxpyKernel
{
public:
    //-----------------------------------------------------------------------------
    //! Vector addition Y = alpha * X + Y.
    //!
    //! \tparam TAcc The type of the accelerator the kernel is executed on..
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator the kernel is executed on.
    //! \param numElements Specifies the number of elements of the vectors X and Y.
    //! \param alpha Scalar the X vector is multiplied with.
    //! \param X Vector of at least n elements.
    //! \param Y Vector of at least n elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TIdx const & numElements,
        TElem const & alpha,
        TElem const * const X,
        TElem * const Y) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The AxpyKernel expects 1-dimensional indices!");

        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            auto const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                Y[i] = alpha * X[i] + Y[i];
            }
        }
    }
};

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<1u>,
    std::size_t>;

TEMPLATE_LIST_TEST_CASE( "axpy", "[axpy]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

#ifdef ALPAKA_CI
    Idx const numElements = 1u<<9u;
#else
    Idx const numElements = 1u<<16u;
#endif

    using Val = float;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;
    using PltfHost = alpaka::pltf::PltfCpu;

    // Create the kernel function object.
    AxpyKernel kernel;

    // Get the host device.
    auto const devHost(
        alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Select a device to execute on.
    auto const devAcc(
        alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    // Get a queue on this device.
    QueueAcc queue(devAcc);

    alpaka::vec::Vec<Dim, Idx> const extent(
        numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            static_cast<Idx>(3u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "AxpyKernel("
        << " numElements:" << numElements
        << ", accelerator: " << alpaka::acc::getAccName<Acc>()
        << ", kernel: " << typeid(kernel).name()
        << ", workDiv: " << workDiv
        << ")" << std::endl;

    // Allocate host memory buffers.
    auto memBufHostX(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));
    auto memBufHostOrigY(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));
    auto memBufHostY(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));
    Val * const pBufHostX = alpaka::mem::view::getPtrNative(memBufHostX);
    Val * const pBufHostOrigY = alpaka::mem::view::getPtrNative(memBufHostOrigY);
    Val * const pBufHostY = alpaka::mem::view::getPtrNative(memBufHostY);

    // random generator for uniformly distributed numbers in [0,1)
    // keep in mind, this can generate different values on different platforms
    std::random_device rd{};
    auto const seed = rd();
    std::default_random_engine eng{ seed };
    std::uniform_real_distribution<Val> dist(0.0, 1.0);
    std::cout << "using seed: " << seed << "\n";
    // Initialize the host input vectors
    for (Idx i(0); i < numElements; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostOrigY[i] = dist(eng);
    }
    Val const alpha( dist(eng) );

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    std::cout << __func__
        << " alpha: " << alpha << std::endl;
    std::cout << __func__ << " X_host: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_host: ";
    alpaka::mem::view::print(memBufHostOrigY, std::cout);
    std::cout << std::endl;
#endif

    // Allocate the buffer on the accelerator.
    auto memBufAccX(alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));
    auto memBufAccY(alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));

    // Copy Host -> Acc.
    alpaka::mem::view::copy(queue, memBufAccX, memBufHostX, extent);
    alpaka::mem::view::copy(queue, memBufAccY, memBufHostOrigY, extent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    alpaka::wait::wait(queue);

    std::cout << __func__ << " X_Dev: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_Dev: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
#endif

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        numElements,
        alpha,
        alpaka::mem::view::getPtrNative(memBufAccX),
        alpaka::mem::view::getPtrNative(memBufAccY)));

    // Profile the kernel execution.
    std::cout << "Execution time: "
        << alpaka::test::integ::measureTaskRunTimeMs(
            queue,
            taskKernel)
        << " ms"
        << std::endl;

    // Copy back the result.
    alpaka::mem::view::copy(queue, memBufHostY, memBufAccY, extent);

    // Wait for the queue to finish the memory operation.
    alpaka::wait::wait(queue);

    bool resultCorrect(true);
    for(Idx i(0u); i < numElements; ++i)
    {
        auto const & val(pBufHostY[i]);
        auto const correctResult(alpha * pBufHostX[i] + pBufHostOrigY[i]);
        auto const relDiff = std::abs((val - correctResult) / std::min(val, correctResult));
        if( relDiff > std::numeric_limits<Val>::epsilon() )
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}
