/* Copyright 2022 Jiří Vyskočil, René Widera, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

// This example generates NUM_ROLLS of random events for each of NUM_POINTS points.
constexpr unsigned NUM_POINTS = 2000; ///< Number of "points". Each will be  processed by a single thread.
constexpr unsigned NUM_ROLLS = 2000; ///< Amount of random number "dice rolls" performed for each "point".

/// Selected PRNG engine
// Comment the current "using" line, and uncomment a different one to change the PRNG engine
template<typename TAcc>
using RandomEngine = alpaka::rand::Philox4x32x10<TAcc>;

// using RandomEngine = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngine = alpaka::rand::engine::cpu::TinyMersenneTwister;
// using RandomEngine = alpaka::rand::engine::uniform_cuda_hip::Xor;


/// Parameters to set up the default accelerator, queue, and buffers
struct Box
{
    // accelerator, queue, and work division typedefs
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using PlatformHost = alpaka::PlatformCpu;
    using Host = alpaka::Dev<PlatformHost>;
    using PlatformAcc = alpaka::Platform<Acc>;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    PlatformHost hostPlatform;
    PlatformAcc accPlatform;
    QueueAcc queue; ///< default accelerator queue

    // buffers holding the PRNG states
    using BufHostRand = alpaka::Buf<Host, RandomEngine<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, RandomEngine<Acc>, Dim, Idx>;

    Vec const extentRand; ///< size of the buffer of PRNG states
    WorkDiv workdivRand; ///< work division for PRNG buffer initialization
    BufHostRand bufHostRand; ///< host side PRNG states buffer (can be used to check the state of the states)
    BufAccRand bufAccRand; ///< device side PRNG states buffer

    // buffers holding the "simulation" results
    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

    Vec const extentResult; ///< size of the results buffer
    WorkDiv workdivResult; ///< work division of the result calculation
    BufHost bufHostResult; ///< host side results buffer
    BufAcc bufAccResult; ///< device side results buffer

    Box()
        : queue{alpaka::getDevByIdx(accPlatform, 0)}
        , extentRand{static_cast<Idx>(NUM_POINTS)} // One PRNG state per "point".
        , workdivRand{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx(accPlatform, 0),
              extentRand,
              Vec(Idx{1}),
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , bufHostRand{alpaka::allocBuf<RandomEngine<Acc>, Idx>(alpaka::getDevByIdx(hostPlatform, 0), extentRand)}
        , bufAccRand{alpaka::allocBuf<RandomEngine<Acc>, Idx>(alpaka::getDevByIdx(accPlatform, 0), extentRand)}
        , extentResult{static_cast<Idx>((NUM_POINTS * NUM_ROLLS))} // Store all "rolls" for each "point"
        , workdivResult{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx(accPlatform, 0),
              extentResult,
              Vec(static_cast<Idx>(NUM_ROLLS)), // One thread per "point"; each performs NUM_ROLLS "rolls"
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , bufHostResult{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx(hostPlatform, 0), extentResult)}
        , bufAccResult{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx(accPlatform, 0), extentResult)}
    {
    }
};

/// PRNG result space division strategy
enum struct Strategy
{
    seed, ///< threads start from different seeds
    subsequence, ///< threads use different subsequences
    offset ///< threads skip a number of elements in the sequence
};

/// Set initial values for the PRNG states. These will be later advanced by the FillKernel
template<Strategy TStrategy>
struct InitRandomKernel
{
};

template<>
struct InitRandomKernel<Strategy::seed>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skipLength = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        if(idx < extent[0])
        {
            TRandEngine engine(idx, 0, 0); // Initialize the engine
            states[idx] = engine; // Save the initial state
        }
    }
};

template<>
struct InitRandomKernel<Strategy::subsequence>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const /* skipLength */ = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        if(idx < extent[0])
        {
            TRandEngine engine(0, idx, 0); // Initialize the engine
            states[idx] = engine; // Save the initial state
        }
    }
};

template<>
struct InitRandomKernel<Strategy::offset>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skipLength = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        if(idx < extent[0])
        {
            TRandEngine engine(0, 0, idx * skipLength); // Initialize the engine
            states[idx] = engine; // Save the initial state
        }
    }
};

/// Fill the result buffer with random "dice rolls"
struct FillKernel
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the results buffer
        RandomEngine<TAcc>* const states, ///< PRNG states buffer
        float* const cells ///< results buffer
    ) const -> void
    {
        /// Index of the current thread. Each thread performs multiple "dice rolls".
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const numGridThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(idx < NUM_POINTS)
        {
            // each worker is handling one random state
            auto const numWorkers
                = alpaka::math::min(acc, numGridThreads, static_cast<decltype(numGridThreads)>(NUM_POINTS));

            RandomEngine<TAcc> engine(states[idx]); // Setup the PRNG using the saved state for this thread.
            alpaka::rand::UniformReal<float> dist; // Setup the random number distribution
            for(uint32_t i = idx; i < extent[0]; i += numWorkers)
            {
                cells[i] = dist(engine); // Roll the dice!
            }
            states[idx] = engine; // Save the final PRNG state
        }
    }
};

/** Save the results to a file and show the calculated average for quick correctness check.
 *
 *  File is in TSV format. One line for each "point"; line length is the number of "rolls".
 */
void saveDataAndShowAverage(std::string filename, float const* buffer, Box const& box)
{
    std::ofstream output(filename);
    std::cout << "Writing " << filename << " ... " << std::flush;
    auto const lineLength = box.extentResult[0] / box.extentRand[0];
    double average = 0;
    for(Box::Idx i = 0; i < box.extentResult[0]; ++i)
    {
        output << buffer[i] << ((i + 1) % lineLength ? "\t" : "\n");
        average += buffer[i];
    }
    average /= box.extentResult[0];
    std::cout << "average value = " << average << " (should be close to 0.5)" << std::endl;
    output.close();
}

template<Strategy TStrategy>
struct Writer;

template<>
struct Writer<Strategy::seed>
{
    static void save(float const* buffer, Box const& box)
    {
        saveDataAndShowAverage("out_seed.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::subsequence>
{
    static void save(float const* buffer, Box const& box)
    {
        saveDataAndShowAverage("out_subsequence.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::offset>
{
    static void save(float const* buffer, Box const& box)
    {
        saveDataAndShowAverage("out_offset.csv", buffer, box);
    }
};

template<Strategy TStrategy>
void runStrategy(Box& box)
{
    // Set up the pointer to the PRNG states buffer
    RandomEngine<Box::Acc>* const ptrBufAccRand{alpaka::getPtrNative(box.bufAccRand)};

    // Initialize the PRNG and its states on the device
    InitRandomKernel<TStrategy> initRandomKernel;
    // The offset strategy needs an additional parameter for initialisation: the offset cannot be deduced form the size
    // of the PRNG buffer and has to be passed in explicitly. Other strategies ignore the last parameter, and deduce
    // the initial parameters solely from the thread index

    alpaka::exec<Box::Acc>(
        box.queue,
        box.workdivRand,
        initRandomKernel,
        box.extentRand,
        ptrBufAccRand,
        static_cast<unsigned>(
            box.extentResult[0] / box.extentRand[0])); // == NUM_ROLLS; amount of work to be performed by each thread

    alpaka::wait(box.queue);

    // OPTIONAL: copy the the initial states to host if you want to check them yourself
    // alpaka_rand::Philox4x32x10<Box::Acc>* const ptrBufHostRand{alpaka::getPtrNative(box.bufHostRand)};
    // alpaka::memcpy(box.queue, box.bufHostRand, box.bufAccRand);
    // alpaka::wait(box.queue);

    // Set up the pointers to the results buffers
    float* const ptrBufHostResult{alpaka::getPtrNative(box.bufHostResult)};
    float* const ptrBufAccResult{alpaka::getPtrNative(box.bufAccResult)};

    // Initialise the results buffer to zero
    for(Box::Idx i = 0; i < box.extentResult[0]; ++i)
        ptrBufHostResult[i] = 0;

    // Run the "computation" kernel filling the results buffer with random numbers in parallel
    alpaka::memcpy(box.queue, box.bufAccResult, box.bufHostResult);
    FillKernel fillKernel;
    alpaka::exec<Box::Acc>(box.queue, box.workdivResult, fillKernel, box.extentResult, ptrBufAccRand, ptrBufAccResult);
    alpaka::memcpy(box.queue, box.bufHostResult, box.bufAccResult);
    alpaka::wait(box.queue);

    // save the results to a CSV file
    Writer<TStrategy>::save(ptrBufHostResult, box);
}

auto main() -> int
{
    Box box; // Initialize the box

    runStrategy<Strategy::seed>(box); // threads start from different seeds
    runStrategy<Strategy::subsequence>(box); // threads use different subsequences
    runStrategy<Strategy::offset>(box); // threads start form an offset equal to the amount of work per thread

    return 0;
}
