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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#define BOOST_TEST_MODULE axpy

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
#include <boost/math/special_functions/relative_difference.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <iostream>
#include <typeinfo>
#include <random>
#include <limits>

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

BOOST_AUTO_TEST_SUITE(axpy)

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<1u>,
    std::size_t>;


BOOST_AUTO_TEST_CASE_TEMPLATE(
    calculateAxpy,
    TAcc,
    TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

#ifdef ALPAKA_CI
    Idx const numElements = 1u<<9u;
#else
    Idx const numElements = 1u<<16u;
#endif

    using Val = float;
    using DevAcc = alpaka::dev::Dev<TAcc>;
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
        alpaka::workdiv::getValidWorkDiv<TAcc>(
            devAcc,
            extent,
            static_cast<Idx>(3u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "AxpyKernel("
        << " numElements:" << numElements
        << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
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

    // C++11 random generator for uniformly distributed numbers in [0,1)
    std::random_device rd{};
    std::default_random_engine eng{ rd() };
    std::uniform_real_distribution<Val> dist(0.0, 1.0);

    // Initialize the host input vectors
    for (Idx i(0); i < numElements; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostOrigY[i] = dist(eng);
    }
    Val const alpha( dist(eng) );

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    std::cout << BOOST_CURRENT_FUNCTION
        << " alpha: " << alpha << std::endl;
    std::cout << BOOST_CURRENT_FUNCTION << " X_host: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << BOOST_CURRENT_FUNCTION << " Y_host: ";
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

    std::cout << BOOST_CURRENT_FUNCTION << " X_Dev: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << BOOST_CURRENT_FUNCTION << " Y_Dev: ";
    alpaka::mem::view::print(memBufHostX, std::cout);
    std::cout << std::endl;
#endif

    // Create the executor task.
    auto const exec(alpaka::kernel::createTaskExec<TAcc>(
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
            exec)
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
        if( boost::math::relative_difference(val, correctResult) > std::numeric_limits<Val>::epsilon() )
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    BOOST_REQUIRE_EQUAL(true, resultCorrect);
}

BOOST_AUTO_TEST_SUITE_END()
