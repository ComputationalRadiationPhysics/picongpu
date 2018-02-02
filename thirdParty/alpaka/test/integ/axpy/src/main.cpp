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
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/stream/Stream.hpp>

#include <iostream>
#include <typeinfo>

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
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TSize const & numElements,
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

            for(TSize i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                Y[i] = alpha * X[i] + Y[i];
            }
        }
    }
};

//#############################################################################
//! Profiles the vector addition kernel.
struct AxpyKernelTester
{
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"  // "comparing floating point with == or != is unsafe"
#endif
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & numElements)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = float;
        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;
        using PltfHost = alpaka::pltf::PltfCpu;

        // Create the kernel function object.
        AxpyKernel kernel;

        // Get the host device.
        auto const devHost(
            alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0u));

        // Get a stream on this device.
        StreamAcc stream(devAcc);

        alpaka::vec::Vec<alpaka::dim::DimInt<1u>, TSize> const extent(
            numElements);

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                extent,
                static_cast<TSize>(3u),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

        std::cout
            << "AxpyKernelTester("
            << " numElements:" << numElements
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate host memory buffers.
        auto memBufHostX(alpaka::mem::buf::alloc<Val, TSize>(devHost, extent));
        auto memBufHostOrigY(alpaka::mem::buf::alloc<Val, TSize>(devHost, extent));
        auto memBufHostY(alpaka::mem::buf::alloc<Val, TSize>(devHost, extent));

        // Initialize the host input vectors
        for (TSize i(0); i < numElements; ++i)
        {
            alpaka::mem::view::getPtrNative(memBufHostX)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
            alpaka::mem::view::getPtrNative(memBufHostOrigY)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        }
        auto const alpha(static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX));

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
        auto memBufAccX(alpaka::mem::buf::alloc<Val, TSize>(devAcc, extent));
        auto memBufAccY(alpaka::mem::buf::alloc<Val, TSize>(devAcc, extent));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(stream, memBufAccX, memBufHostX, extent);
        alpaka::mem::view::copy(stream, memBufAccY, memBufHostOrigY, extent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        alpaka::wait::wait(stream);

        std::cout << BOOST_CURRENT_FUNCTION << " X_Dev: ";
        alpaka::mem::view::print(memBufHostX, std::cout);
        std::cout << std::endl;
        std::cout << BOOST_CURRENT_FUNCTION << " Y_Dev: ";
        alpaka::mem::view::print(memBufHostX, std::cout);
        std::cout << std::endl;
#endif

        // Create the executor task.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            numElements,
            alpha,
            alpaka::mem::view::getPtrNative(memBufAccX),
            alpaka::mem::view::getPtrNative(memBufAccY)));

        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::test::integ::measureKernelRunTimeMs(
                stream,
                exec)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(stream, memBufHostY, memBufAccY, extent);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        bool resultCorrect(true);
        auto const pHostResultData(alpaka::mem::view::getPtrNative(memBufHostY));
        for(TSize i(0u);
            i < numElements;
            ++i)
        {
            auto const & val(pHostResultData[i]);
            auto const correctResult(alpha * alpaka::mem::view::getPtrNative(memBufHostX)[i] + alpaka::mem::view::getPtrNative(memBufHostOrigY)[i]);
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#endif
            if(val != correctResult)
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
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

        std::cout << "################################################################################" << std::endl;

        allResultsCorrect = allResultsCorrect && resultCorrect;
    }
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

public:
    bool allResultsCorrect = true;
};

auto main()
-> int
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                                alpaka axpy test                                " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::test::acc::writeEnabledAccs<alpaka::dim::DimInt<1u>, std::size_t>(std::cout);

        std::cout << std::endl;

        AxpyKernelTester axpyKernelTester;

        // For different sizes.
#ifdef ALPAKA_CI
        for(std::size_t vecSize(1u); vecSize <= 1u<<9u; vecSize *= 8u)
#else
        for(std::size_t vecSize(1u); vecSize <= 1u<<16u; vecSize *= 2u)
#endif
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            alpaka::meta::forEachType<
                alpaka::test::acc::EnabledAccs<alpaka::dim::DimInt<1u>, std::size_t>>(
                    axpyKernelTester,
                    vecSize);
        }
        return axpyKernelTester.allResultsCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch(std::exception const & e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unknown Exception" << std::endl;
        return EXIT_FAILURE;
    }
}
