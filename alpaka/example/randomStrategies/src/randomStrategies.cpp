/* Copyright 2022 Jiří Vyskočil, René Widera, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

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
using RandomEngine = alpaka::rand::Philox4x32x10;

// using RandomEngine = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngine = alpaka::rand::engine::cpu::TinyMersenneTwister;
// using RandomEngine = alpaka::rand::engine::uniform_cuda_hip::Xor;


/// Parameters to set up the default accelerator, queue, and buffers
template<typename TAccTag>
struct Box
{
    // accelerator, queue, and work division typedefs
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
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
    using BufHostRand = alpaka::Buf<Host, RandomEngine, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, RandomEngine, Dim, Idx>;

    Vec const extentRand; ///< size of the buffer of PRNG states
    // WorkDiv workdivRand; ///< work division for PRNG buffer initialization // REMOVE THAT!!
    // WorkDiv workdivResult; ///< work division of the result calculation // REMOVE THAT!!
    BufHostRand bufHostRand; ///< host side PRNG states buffer (can be used to check the state of the states)
    BufAccRand bufAccRand; ///< device side PRNG states buffer

    // buffers holding the "simulation" results
    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

    Vec const extentResult; ///< size of the results buffer

    BufHost bufHostResult; ///< host side results buffer
    BufAcc bufAccResult; ///< device side results buffer

    Box()
        : queue{alpaka::getDevByIdx(accPlatform, 0)}
        , extentRand{static_cast<Idx>(NUM_POINTS)} // One PRNG state per "point".
        , bufHostRand{alpaka::allocBuf<RandomEngine, Idx>(alpaka::getDevByIdx(hostPlatform, 0), extentRand)}
        , bufAccRand{alpaka::allocBuf<RandomEngine, Idx>(alpaka::getDevByIdx(accPlatform, 0), extentRand)}
        , extentResult{static_cast<Idx>((NUM_POINTS * NUM_ROLLS))} // Store all "rolls" for each "point"
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
        RandomEngine* const states, ///< PRNG states buffer
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

            RandomEngine engine(states[idx]); // Setup the PRNG using the saved state for this thread.
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
template<typename TAccTag>
void saveDataAndShowAverage(std::string filename, float const* buffer, Box<TAccTag> const& box)
{
    std::ofstream output(filename);
    std::cout << "Writing " << filename << " ... " << std::flush;
    auto const lineLength = box.extentResult[0] / box.extentRand[0];
    double average = 0;
    for(typename Box<TAccTag>::Idx i = 0; i < box.extentResult[0]; ++i)
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
    template<typename TAccTag>
    static void save(float const* buffer, Box<TAccTag> const& box)
    {
        saveDataAndShowAverage("out_seed.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::subsequence>
{
    template<typename TAccTag>
    static void save(float const* buffer, Box<TAccTag> const& box)
    {
        saveDataAndShowAverage("out_subsequence.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::offset>
{
    template<typename TAccTag>
    static void save(float const* buffer, Box<TAccTag> const& box)
    {
        saveDataAndShowAverage("out_offset.csv", buffer, box);
    }
};

template<Strategy TStrategy, typename TAccTag>
void runStrategy(Box<TAccTag>& box)
{
    // Set up the pointer to the PRNG states buffer
    RandomEngine* const ptrBufAccRand{std::data(box.bufAccRand)};

    // Initialize the PRNG and its states on the device
    InitRandomKernel<TStrategy> initRandomKernel;
    // The offset strategy needs an additional parameter for initialisation: the offset cannot be deduced form the size
    // of the PRNG buffer and has to be passed in explicitly. Other strategies ignore the last parameter, and deduce
    // the initial parameters solely from the thread index


    alpaka::KernelCfg<typename Box<TAccTag>::Acc> kernelCfg
        = {box.extentRand,
           typename Box<TAccTag>::Vec(typename Box<TAccTag>::Idx{1}),
           false,
           alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDivRand = alpaka::getValidWorkDiv(
        kernelCfg,
        alpaka::getDevByIdx(box.accPlatform, 0),
        initRandomKernel,
        box.extentRand,
        ptrBufAccRand,
        static_cast<unsigned>(box.extentResult[0] / box.extentRand[0]));


    alpaka::exec<typename Box<TAccTag>::Acc>(
        box.queue,
        workDivRand,
        initRandomKernel,
        box.extentRand,
        ptrBufAccRand,
        static_cast<unsigned>(
            box.extentResult[0] / box.extentRand[0])); // == NUM_ROLLS; amount of work to be performed by each thread

    alpaka::wait(box.queue);

    // OPTIONAL: copy the the initial states to host if you want to check them yourself
    // alpaka_rand::Philox4x32x10<Box::Acc>* const ptrBufHostRand{std::data(box.bufHostRand)};
    // alpaka::memcpy(box.queue, box.bufHostRand, box.bufAccRand);
    // alpaka::wait(box.queue);

    // Set up the pointers to the results buffers
    float* const ptrBufHostResult{std::data(box.bufHostResult)};
    float* const ptrBufAccResult{std::data(box.bufAccResult)};

    // Initialise the results buffer to zero
    for(typename Box<TAccTag>::Idx i = 0; i < box.extentResult[0]; ++i)
        ptrBufHostResult[i] = 0;

    // Run the "computation" kernel filling the results buffer with random numbers in parallel
    alpaka::memcpy(box.queue, box.bufAccResult, box.bufHostResult);
    FillKernel fillKernel;

    alpaka::KernelCfg<typename Box<TAccTag>::Acc> fillKernelCfg
        = {box.extentResult,
           typename Box<TAccTag>::Vec(static_cast<typename Box<TAccTag>::Idx>(
               NUM_ROLLS)), // One thread per "point"; each performs NUM_ROLLS "rolls"
           false,
           alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workdivResult = alpaka::getValidWorkDiv(
        fillKernelCfg,
        alpaka::getDevByIdx(box.accPlatform, 0),
        fillKernel,
        box.extentResult,
        ptrBufAccRand,
        ptrBufAccResult);


    alpaka::exec<typename Box<TAccTag>::Acc>(
        box.queue,
        workdivResult,
        fillKernel,
        box.extentResult,
        ptrBufAccRand,
        ptrBufAccResult);
    alpaka::memcpy(box.queue, box.bufHostResult, box.bufAccResult);
    alpaka::wait(box.queue);

    // save the results to a CSV file
    Writer<TStrategy>::save(ptrBufHostResult, box);
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    Box<TAccTag> box; // Initialize the box

    runStrategy<Strategy::seed>(box); // threads start from different seeds
    runStrategy<Strategy::subsequence>(box); // threads use different subsequences
    runStrategy<Strategy::offset>(box); // threads start form an offset equal to the amount of work per thread

    return 0;
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
