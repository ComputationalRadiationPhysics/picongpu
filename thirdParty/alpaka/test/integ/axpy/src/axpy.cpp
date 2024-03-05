/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <typeinfo>

//! A vector addition kernel.
class AxpyKernel
{
public:
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
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIdx const& numElements,
        TElem const& alpha,
        TElem const* const X,
        TElem* const Y) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The AxpyKernel expects 1-dimensional indices!");

        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        auto const threadElemExtent = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u];
        auto const threadFirstElemIdx = gridThreadIdx * threadElemExtent;

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx = threadFirstElemIdx + threadElemExtent;
            auto const threadLastElemIdxClipped = (numElements > threadLastElemIdx) ? threadLastElemIdx : numElements;

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                Y[i] = alpha * X[i] + Y[i];
            }
        }
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("axpy", "[axpy]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

#ifdef ALPAKA_CI
    Idx const numElements = 1u << 9u;
#else
    Idx const numElements = 1u << 16u;
#endif

    using Val = float;
    using DevAcc = alpaka::Dev<Acc>;
    using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;

    // Create the kernel function object.
    AxpyKernel kernel;

    // Get the host device.
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Select a device to execute on.
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Get a queue on this device.
    QueueAcc queue(devAcc);

    alpaka::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        static_cast<Idx>(3u),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "AxpyKernel("
              << " numElements:" << numElements << ", accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << alpaka::core::demangled<decltype(kernel)> << ", workDiv: " << workDiv << ")"
              << std::endl;

    // Allocate host memory buffers in pinned memory.
    auto memBufHostX = alpaka::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostOrigY = alpaka::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostY = alpaka::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    Val* const pBufHostX = alpaka::getPtrNative(memBufHostX);
    Val* const pBufHostOrigY = alpaka::getPtrNative(memBufHostOrigY);
    Val* const pBufHostY = alpaka::getPtrNative(memBufHostY);

    // random generator for uniformly distributed numbers in [0,1)
    // keep in mind, this can generate different values on different platforms
    std::random_device rd{};
    auto const seed = rd();
    std::default_random_engine eng{seed};
    std::uniform_real_distribution<Val> dist(Val{0}, Val{1});
    std::cout << "using seed: " << seed << "\n";
    // Initialize the host input vectors
    for(Idx i(0); i < numElements; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostOrigY[i] = dist(eng);
    }
    Val const alpha(dist(eng));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    std::cout << __func__ << " alpha: " << alpha << std::endl;
    std::cout << __func__ << " X_host: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_host: ";
    alpaka::print(memBufHostOrigY, std::cout);
    std::cout << std::endl;
#endif

    // Allocate the buffer on the accelerator.
    auto memBufAccX = alpaka::allocBuf<Val, Idx>(devAcc, extent);
    auto memBufAccY = alpaka::allocBuf<Val, Idx>(devAcc, extent);

    // Copy Host -> Acc.
    alpaka::memcpy(queue, memBufAccX, memBufHostX);
    alpaka::memcpy(queue, memBufAccY, memBufHostOrigY);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    alpaka::wait(queue);

    std::cout << __func__ << " X_Dev: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_Dev: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
#endif

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        numElements,
        alpha,
        alpaka::getPtrNative(memBufAccX),
        alpaka::getPtrNative(memBufAccY));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue, memBufHostY, memBufAccY);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue);

    bool resultCorrect(true);
    for(Idx i(0u); i < numElements; ++i)
    {
        auto const& val(pBufHostY[i]);
        auto const correctResult = alpha * pBufHostX[i] + pBufHostOrigY[i];
        auto const relDiff = std::abs((val - correctResult) / std::min(val, correctResult));
        if(relDiff > std::numeric_limits<Val>::epsilon())
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}
