/* Copyright 2021 Jiri Vyskocil
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <tuple>

constexpr unsigned NUM_CALCULATIONS = 256;
constexpr unsigned NUM_X = 127;
constexpr unsigned NUM_Y = 211;

/// Selected PRNG engine for single-value operation
template<typename TAcc>
using RandomEngineSingle = alpaka::rand::Philox4x32x10<TAcc>;
// using RandomEngineSingle = alpaka::rand::engine::uniform_cuda_hip::Xor;
// using RandomEngineSingle = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngineSingle = alpaka::rand::engine::cpu::TinyMersenneTwister;


/// Selected PRNG engine for vector operation
template<typename TAcc>
using RandomEngineVector = alpaka::rand::Philox4x32x10Vector<TAcc>;

/** Get a  pointer to the correct location of `TElement array` taking pitch into account.
 *
 *  The pitch might not be a multiple of `sizeof(TElement)`, especially in the case of random generator state which can
 *  be any size. Therefore we cast the array to `std::byte*`, do the pointer offset calculation with byte precision,
 *  and cast the resulting location back to TElement*.
 */
template<typename TElement, typename TIndex>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto pitchedPointer2D(TElement* const array, std::size_t pitch, TIndex idx)
    -> TElement* const
{
    // `idx[0]` is the index of the row - consecutive rows are `pitch` bytes away from each other. `idx[1]` is the
    // index of the element in the given row - consecutive elements are `sizeof(TElement)` away from each other.
    // Potential inner alignment should be already included in `sizeof(TElement)`.
    auto const memoryLocationIdx = idx[0] * pitch + idx[1] * sizeof(TElement);
    std::byte* bytePointer = reinterpret_cast<std::byte*>(array) + memoryLocationIdx;
    return reinterpret_cast<TElement*>(bytePointer);
}

struct InitRandomKernel
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        TRandEngine* const states,
        std::size_t pitchRand) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        if(idx[0] < NUM_Y && idx[1] < NUM_X)
        {
            auto const linearIdx = alpaka::mapIdx<1u>(idx, extent)[0];
            auto statesOut = pitchedPointer2D(states, pitchRand, idx);
            TRandEngine engine(42, static_cast<std::uint32_t>(linearIdx));
            *statesOut = engine;
        }
    }
};

struct RunTimestepKernelSingle
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        RandomEngineSingle<TAcc>* const states,
        float* const cells,
        std::size_t pitchRand,
        std::size_t pitchOut) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        if(idx[0] < NUM_Y && idx[1] < NUM_X)
        {
            auto statesOut = pitchedPointer2D(states, pitchRand, idx);
            auto cellsOut = pitchedPointer2D(cells, pitchOut, idx);

            // Setup generator and distribution.
            RandomEngineSingle<TAcc> engine(*statesOut);
            alpaka::rand::UniformReal<float> dist;

            float sum = 0;
            for(unsigned numCalculations = 0; numCalculations < NUM_CALCULATIONS; ++numCalculations)
            {
                sum += dist(engine);
            }
            *cellsOut = sum / NUM_CALCULATIONS;
            *statesOut = engine;
        }
    }
};

struct RunTimestepKernelVector
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        RandomEngineVector<TAcc>* const states,
        float* const cells,
        std::size_t pitchRand,
        std::size_t pitchOut) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        if(idx[0] < NUM_Y && idx[1] < NUM_X)
        {
            auto statesOut = pitchedPointer2D(states, pitchRand, idx);
            auto cellsOut = pitchedPointer2D(cells, pitchOut, idx);

            // Setup generator and distribution.
            RandomEngineVector<TAcc> engine(*statesOut); // Load the state of the random engine
            using DistributionResult =
                typename RandomEngineVector<TAcc>::template ResultContainer<float>; // Container type which will store
                                                                                    // the distribution results
            constexpr unsigned resultVectorSize = std::tuple_size_v<DistributionResult>; // Size of the result vector
            alpaka::rand::UniformReal<DistributionResult> dist; // Vector-aware distribution function


            float sum = 0;
            static_assert(
                NUM_CALCULATIONS % resultVectorSize == 0,
                "Number of calculations must be a multiple of result vector size.");
            for(unsigned numCalculations = 0; numCalculations < NUM_CALCULATIONS / resultVectorSize; ++numCalculations)
            {
                auto result = dist(engine);
                for(unsigned i = 0; i < resultVectorSize; ++i)
                {
                    sum += result[i];
                }
            }
            *cellsOut = sum / NUM_CALCULATIONS;
            *statesOut = engine;
        }
    }
};

auto main() -> int
{
    using Dim = alpaka::DimInt<2>;
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

    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;
    using BufHostRand = alpaka::Buf<Host, RandomEngineSingle<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, RandomEngineSingle<Acc>, Dim, Idx>;
    using BufHostRandVec = alpaka::Buf<Host, RandomEngineVector<Acc>, Dim, Idx>;
    using BufAccRandVec = alpaka::Buf<Acc, RandomEngineVector<Acc>, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    constexpr Idx numX = NUM_X;
    constexpr Idx numY = NUM_Y;

    const Vec extent(numY, numX);

    constexpr Idx perThreadX = 1;
    constexpr Idx perThreadY = 1;

    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        Vec(perThreadY, perThreadX),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Setup buffer.
    BufHost bufHostS{alpaka::allocBuf<float, Idx>(devHost, extent)};
    float* const ptrBufHostS{alpaka::getPtrNative(bufHostS)};
    BufAcc bufAccS{alpaka::allocBuf<float, Idx>(devAcc, extent)};
    float* const ptrBufAccS{alpaka::getPtrNative(bufAccS)};

    BufHost bufHostV{alpaka::allocBuf<float, Idx>(devHost, extent)};
    float* const ptrBufHostV{alpaka::getPtrNative(bufHostV)};
    BufAcc bufAccV{alpaka::allocBuf<float, Idx>(devAcc, extent)};
    float* const ptrBufAccV{alpaka::getPtrNative(bufAccV)};

    BufHostRand bufHostRandS{alpaka::allocBuf<RandomEngineSingle<Acc>, Idx>(devHost, extent)};
    BufAccRand bufAccRandS{alpaka::allocBuf<RandomEngineSingle<Acc>, Idx>(devAcc, extent)};
    RandomEngineSingle<Acc>* const ptrBufAccRandS{alpaka::getPtrNative(bufAccRandS)};

    BufHostRandVec bufHostRandV{alpaka::allocBuf<RandomEngineVector<Acc>, Idx>(devHost, extent)};
    BufAccRandVec bufAccRandV{alpaka::allocBuf<RandomEngineVector<Acc>, Idx>(devAcc, extent)};
    RandomEngineVector<Acc>* const ptrBufAccRandV{alpaka::getPtrNative(bufAccRandV)};

    InitRandomKernel initRandomKernel;
    auto pitchBufAccRandS = alpaka::getPitchesInBytes(bufAccRandS)[0];
    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandS, pitchBufAccRandS);
    alpaka::wait(queue);

    auto pitchBufAccRandV = alpaka::getPitchesInBytes(bufAccRandV)[0];
    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandV, pitchBufAccRandV);
    alpaka::wait(queue);

    auto pitchHostS = alpaka::getPitchesInBytes(bufHostS)[0];
    auto pitchHostV = alpaka::getPitchesInBytes(bufHostV)[0];

    for(Idx y = 0; y < numY; ++y)
    {
        for(Idx x = 0; x < numX; ++x)
        {
            *pitchedPointer2D(ptrBufHostS, pitchHostS, Vec(y, x)) = 0;
            *pitchedPointer2D(ptrBufHostV, pitchHostV, Vec(y, x)) = 0;
        }
    }

    auto pitchBufAccS = alpaka::getPitchesInBytes(bufAccS)[0];
    alpaka::memcpy(queue, bufAccS, bufHostS);
    RunTimestepKernelSingle runTimestepKernelSingle;
    alpaka::exec<Acc>(
        queue,
        workdiv,
        runTimestepKernelSingle,
        extent,
        ptrBufAccRandS,
        ptrBufAccS,
        pitchBufAccRandS,
        pitchBufAccS);
    alpaka::memcpy(queue, bufHostS, bufAccS);

    auto pitchBufAccV = alpaka::getPitchesInBytes(bufAccV)[0];
    alpaka::memcpy(queue, bufAccV, bufHostV);
    RunTimestepKernelVector runTimestepKernelVector;
    alpaka::exec<Acc>(
        queue,
        workdiv,
        runTimestepKernelVector,
        extent,
        ptrBufAccRandV,
        ptrBufAccV,
        pitchBufAccRandV,
        pitchBufAccV);
    alpaka::memcpy(queue, bufHostV, bufAccV);
    alpaka::wait(queue);

    float avgS = 0;
    float avgV = 0;
    for(Idx y = 0; y < numY; ++y)
    {
        for(Idx x = 0; x < numX; ++x)
        {
            avgS += *pitchedPointer2D(ptrBufHostS, pitchHostS, Vec(y, x));
            avgV += *pitchedPointer2D(ptrBufHostV, pitchHostV, Vec(y, x));
        }
    }
    avgS /= numX * numY;
    avgV /= numX * numY;

    auto totalCalculations = numX * numY * NUM_CALCULATIONS;
    float expectedValue = 0.5f;
    std::cout << "Number of cells: " << numX * numY << "\n";
    std::cout << "Number of calculations per cell: " << NUM_CALCULATIONS << "\n";
    std::cout << "Total number of calculations: " << totalCalculations << "\n";
    std::cout << "Mean value A: " << avgS << " (should converge to " << expectedValue << ")\n";
    std::cout << "Mean value B: " << avgV << " (should converge to " << expectedValue << ")\n";

    float convergenceFactor = expectedValue / std::sqrt(totalCalculations);
    std::cout << "Maximum error expected at " << totalCalculations << " calculations should be around "
              << convergenceFactor << std::endl;
    // 10 is a magic number to allow a reasonable margin statistical errors
    if(std::abs(avgS - expectedValue) < 10 * convergenceFactor
       || std::abs(avgV - expectedValue) < 10 * convergenceFactor)
    {
        std::cout << "Convergence test passed" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Convergence test failed!" << std::endl;
        return 1;
    }
}
