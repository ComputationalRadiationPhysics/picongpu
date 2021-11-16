/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <tuple>

unsigned constexpr NUM_CALCULATIONS = 256;
unsigned constexpr NUM_X = 1237;
unsigned constexpr NUM_Y = 2131;

/// Selected PRNG engine for single-value operation
template<typename TAcc>
using RandomEngineSingle = alpaka::rand::Philox4x32x10<TAcc>;
// using RandomEngineSingle = alpaka::rand::engine::uniform_cuda_hip::Xor;
// using RandomEngineSingle = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngineSingle = alpaka::rand::engine::cpu::TinyMersenneTwister;


/// Selected PRNG engine for vector operation
template<typename TAcc>
using RandomEngineVector = alpaka::rand::Philox4x32x10Vector<TAcc>;

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
        auto const linearIdx = alpaka::mapIdx<1u>(idx, extent)[0];
        auto const memoryLocationIdx = idx[0] * pitchRand + idx[1];
        TRandEngine engine(42, static_cast<std::uint32_t>(linearIdx));
        states[memoryLocationIdx] = engine;
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
        auto const memoryLocationRandIdx = idx[0] * pitchRand + idx[1];
        auto const memoryLocationOutIdx = idx[0] * pitchOut + idx[1];

        // Setup generator and distribution.
        RandomEngineSingle<TAcc> engine(states[memoryLocationRandIdx]);
        alpaka::rand::UniformReal<float> dist;

        float sum = 0;
        for(unsigned numCalculations = 0; numCalculations < NUM_CALCULATIONS; ++numCalculations)
        {
            sum += dist(engine);
        }
        cells[memoryLocationOutIdx] = sum / NUM_CALCULATIONS;
        states[memoryLocationRandIdx] = engine;
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
        auto const memoryLocationRandIdx = idx[0] * pitchRand + idx[1];
        auto const memoryLocationOutIdx = idx[0] * pitchOut + idx[1];

        // Setup generator and distribution.
        RandomEngineVector<TAcc> engine(states[memoryLocationRandIdx]); // Load the state of the random engine
        using DistributionResult =
            typename RandomEngineVector<TAcc>::template ResultContainer<float>; // Container type which will store the
                                                                                // distribution results
        unsigned constexpr resultVectorSize = std::tuple_size<DistributionResult>::value; // Size of the result vector
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
        cells[memoryLocationOutIdx] = sum / NUM_CALCULATIONS;
        states[memoryLocationRandIdx] = engine;
    }
};

auto main() -> int
{
    using Dim = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<Host>(0u);
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
    auto pitchBufAccRandS = alpaka::getPitchBytes<1u>(bufAccRandS) / sizeof(RandomEngineSingle<Acc>);
    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandS, pitchBufAccRandS);
    alpaka::wait(queue);

    auto pitchBufAccRandV = alpaka::getPitchBytes<1u>(bufAccRandV) / sizeof(RandomEngineVector<Acc>);
    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandV, pitchBufAccRandV);
    alpaka::wait(queue);

    auto pitchHostS = alpaka::getPitchBytes<1u>(bufHostS) / sizeof(float); /// \todo: get the type from bufHostS
    auto pitchHostV = alpaka::getPitchBytes<1u>(bufHostV) / sizeof(float); /// \todo: get the type from bufHostV

    for(Idx y = 0; y < numY; ++y)
    {
        for(Idx x = 0; x < numX; ++x)
        {
            ptrBufHostS[y * pitchHostS + x] = 0;
            ptrBufHostV[y * pitchHostV + x] = 0;
        }
    }

    /// \todo get the types from respective function parameters
    auto pitchBufAccS = alpaka::getPitchBytes<1u>(bufAccS) / sizeof(float);
    alpaka::memcpy(queue, bufAccS, bufHostS, extent);
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
    alpaka::memcpy(queue, bufHostS, bufAccS, extent);

    auto pitchBufAccV = alpaka::getPitchBytes<1u>(bufAccV) / sizeof(float);
    alpaka::memcpy(queue, bufAccV, bufHostV, extent);
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
    alpaka::memcpy(queue, bufHostV, bufAccV, extent);
    alpaka::wait(queue);

    float avgS = 0;
    float avgV = 0;
    for(Idx y = 0; y < numY; ++y)
    {
        for(Idx x = 0; x < numX; ++x)
        {
            avgS += ptrBufHostS[y * pitchHostS + x];
            avgV += ptrBufHostV[y * pitchHostV + x];
        }
    }
    avgS /= numX * numY;
    avgV /= numX * numY;

    std::cout << "Number of cells: " << numX * numY << "\n";
    std::cout << "Number of calculations: " << NUM_CALCULATIONS << "\n";
    std::cout << "Mean value A: " << avgS << " (should converge to 0.5)\n";
    std::cout << "Mean value B: " << avgV << " (should converge to 0.5)\n";

    return 0;
}
