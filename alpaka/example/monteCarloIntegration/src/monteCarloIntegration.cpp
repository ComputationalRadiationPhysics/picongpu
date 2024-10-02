/* Copyright 2020 Benjamin Worpitz, Sergei Bastrakov, Jakob Krude, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

//! This functor defines the function for which the integral is to be computed.
struct Function
{
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param x The argument.
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, float const x) -> float
    {
        return alpaka::math::sqrt(acc, (1.0f - x * x));
    }
};

//! The kernel executing the parallel logic.
//! Each Thread generates X pseudo random numbers and compares them with the given function.
//! The local result will be added to a global result.
struct Kernel
{
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TFunctor A wrapper for a function.
    //! \param acc The accelerator to be executed on.
    //! \param numPoints The total number of points to be calculated.
    //! \param globalCounter The sum of all local results.
    //! \param functor The function for which the integral is to be computed.
    template<typename TAcc, typename TFunctor>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        size_t const numPoints,
        uint32_t* const globalCounter,
        TFunctor functor) const -> void
    {
        // Get the global linearized thread idx.
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];
        // Setup generator engine and distribution.
        auto engine = alpaka::rand::engine::createDefault(
            acc,
            linearizedGlobalThreadIdx,
            0); // No specific subsequence start.
        // For simplicity the interval is fixed to [0.0,1.0].
        auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));

        uint32_t localCount = 0;
        for(size_t i = linearizedGlobalThreadIdx; i < numPoints; i += globalThreadExtent.prod())
        {
            // Generate a point in the 2D interval.
            float x = dist(engine);
            float y = dist(engine);
            // Count every time where the point is "below" the given function.
            if(y <= functor(acc, x))
            {
                ++localCount;
            }
        }

        // Add the local result to the sum of the other results.
        alpaka::atomicAdd(acc, globalCounter, localCount, alpaka::hierarchy::Blocks{});
    }
};

auto main() -> int
{
    // Defines and setup.
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

    using BufHost = alpaka::Buf<Host, uint32_t, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, uint32_t, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    // Problem parameter.
    constexpr size_t numPoints = 1'000'000u;
    constexpr size_t extent = 1u;
    constexpr size_t numThreads = 100u; // Kernel will decide numCalcPerThread.
    constexpr size_t numAlpakaElementsPerThread = 1;
    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        devAcc,
        Vec(numThreads),
        Vec(numAlpakaElementsPerThread),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Setup buffer.
    BufHost bufHost{alpaka::allocBuf<uint32_t, Idx>(devHost, extent)};
    uint32_t* const ptrBufHost{alpaka::getPtrNative(bufHost)};
    BufAcc bufAcc{alpaka::allocBuf<uint32_t, Idx>(devAcc, extent)};
    uint32_t* const ptrBufAcc{alpaka::getPtrNative(bufAcc)};

    // Initialize the global count to 0.
    ptrBufHost[0] = 0.0f;
    alpaka::memcpy(queue, bufAcc, bufHost);

    Kernel kernel;
    alpaka::exec<Acc>(queue, workdiv, kernel, numPoints, ptrBufAcc, Function{});
    alpaka::memcpy(queue, bufHost, bufAcc);
    alpaka::wait(queue);

    // Check the result.
    uint32_t globalCount = *ptrBufHost;

    // Final result.
    float finalResult = globalCount / static_cast<float>(numPoints);
    constexpr double pi = 3.14159265358979323846;
    constexpr double exactResult = pi / 4.0;
    auto const error = std::abs(finalResult - exactResult);

    std::cout << "exact result (pi / 4): " << pi / 4.0 << "\n";
    std::cout << "final result: " << finalResult << "\n";
    std::cout << "error: " << error << "\n";
    return error > 0.001 ? EXIT_FAILURE : EXIT_SUCCESS;
}
