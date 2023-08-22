/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <typeinfo>
#include <vector>

//! A kernel using atomicOp, syncBlockThreads, getDynSharedMem, getIdx, getWorkDiv and global memory to compute a
//! (useless) result. \tparam TnumUselessWork The number of useless calculations done in each kernel execution.
template<typename TnumUselessWork, typename Val>
class SharedMemKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, Val* const puiBlockRetVals) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        static_assert(alpaka::Dim<TAcc>::value == 1, "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        Idx const blockThreadCount(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // Get the dynamically allocated shared memory.
        Val* const pBlockShared(alpaka::getDynSharedMem<Val>(acc));

        // Calculate linearized index of the thread in the block.
        Idx const blockThreadIdx1d(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        auto sum1 = static_cast<Val>(blockThreadIdx1d + 1);
        for(Val i(0); i < static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum1 += i;
        }
        pBlockShared[blockThreadIdx1d] = sum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        alpaka::syncBlockThreads(acc);

        // Do something useless.
        auto sum2 = static_cast<Val>(blockThreadIdx1d);
        for(Val i(0); i < static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        pBlockShared[(blockThreadCount - 1) - blockThreadIdx1d] += sum2;


        // Synchronize all threads again.
        alpaka::syncBlockThreads(acc);

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(blockThreadIdx1d > 0)
        {
            alpaka::atomicAdd(acc, &pBlockShared[0], pBlockShared[blockThreadIdx1d]);
        }


        alpaka::syncBlockThreads(acc);

        // Only master writes result to global memory.
        if(blockThreadIdx1d == 0)
        {
            // Calculate linearized block id.
            Idx const gridBlockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

            puiBlockRetVals[gridBlockIdx] = pBlockShared[0];
        }
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TnumUselessWork, typename Val, typename TAcc>
    struct BlockSharedMemDynSizeBytes<SharedMemKernel<TnumUselessWork, Val>, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec, typename... TArgs>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            SharedMemKernel<TnumUselessWork, Val> const& /* sharedMemKernel */,
            TVec const& blockThreadExtent,
            TVec const& threadElemExtent,
            TArgs&&...) -> std::size_t
        {
            return static_cast<std::size_t>(blockThreadExtent.prod() * threadElemExtent.prod()) * sizeof(Val);
        }
    };
} // namespace alpaka::trait

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("sharedMem", "[sharedMem]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Idx const numElements = 1u << 16u;

    using Val = std::int32_t;
    using TnumUselessWork = std::integral_constant<Idx, 100>;

    using DevAcc = alpaka::Dev<Acc>;
    using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;


    // Create the kernel function object.
    SharedMemKernel<TnumUselessWork, Val> kernel;

    // Select a device to execute on.
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Get a queue on this device.
    QueueAcc queue(devAcc);

    // Set the grid blocks extent.
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        numElements,
        static_cast<Idx>(1u),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "SharedMemKernel("
              << " accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << alpaka::core::demangled<decltype(kernel)> << ", workDiv: " << workDiv << ")"
              << std::endl;

    Idx const gridBlocksCount(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[0u]);
    Idx const blockThreadCount(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[0u]);

    // An array for the return values calculated by the blocks.
    std::vector<Val> blockRetVals(static_cast<std::size_t>(gridBlocksCount));

    // Allocate accelerator buffers and copy.
    Idx const resultElemCount(gridBlocksCount);
    auto blockRetValsAcc = alpaka::allocBuf<Val, Idx>(devAcc, resultElemCount);
    alpaka::memcpy(queue, blockRetValsAcc, blockRetVals, resultElemCount);

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, kernel, alpaka::getPtrNative(blockRetValsAcc));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue, blockRetVals, blockRetValsAcc, resultElemCount);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue);

    // Assert that the results are correct.
    Val const correctResult(static_cast<Val>(blockThreadCount * blockThreadCount));

    bool resultCorrect(true);
    for(Idx i(0); i < gridBlocksCount; ++i)
    {
        auto const val(blockRetVals[static_cast<std::size_t>(i)]);
        if(val != correctResult)
        {
            std::cerr << "blockRetVals[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}
