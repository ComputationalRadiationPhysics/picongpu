/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
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
#include <vector>

//#############################################################################
//! A kernel using atomicOp, syncBlockThreads, getMem, getIdx, getWorkDiv and global memory to compute a (useless) result.
//! \tparam TnumUselessWork The number of useless calculations done in each kernel execution.
template<
    typename TnumUselessWork,
    typename Val>
class SharedMemKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        Val * const puiBlockRetVals) const
    -> void
    {
        using Idx = alpaka::idx::Idx<TAcc>;

        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        Idx const blockThreadCount(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // Get the dynamically allocated shared memory.
        Val * const pBlockShared(alpaka::block::shared::dyn::getMem<Val>(acc));

        // Calculate linearized index of the thread in the block.
        Idx const blockThreadIdx1d(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        auto sum1 = static_cast<Val>(blockThreadIdx1d+1);
        for(Val i(0); i<static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum1 += i;
        }
        pBlockShared[blockThreadIdx1d] = sum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        alpaka::block::sync::syncBlockThreads(acc);

        // Do something useless.
        auto sum2 = static_cast<Val>(blockThreadIdx1d);
        for(Val i(0); i<static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        pBlockShared[(blockThreadCount-1)-blockThreadIdx1d] += sum2;


        // Synchronize all threads again.
        alpaka::block::sync::syncBlockThreads(acc);

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(blockThreadIdx1d > 0)
        {
            alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &pBlockShared[0], pBlockShared[blockThreadIdx1d]);
        }


        alpaka::block::sync::syncBlockThreads(acc);

        // Only master writes result to global memory.
        if(blockThreadIdx1d==0)
        {
            // Calculate linearized block id.
            Idx const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

            puiBlockRetVals[gridBlockIdx] = pBlockShared[0];
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            template<
                typename TnumUselessWork,
                typename Val,
                typename TAcc>
            struct BlockSharedMemDynSizeBytes<
                SharedMemKernel<TnumUselessWork, Val>,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    SharedMemKernel<TnumUselessWork, Val> const & sharedMemKernel,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    TArgs && ...)
                -> idx::Idx<TAcc>
                {
                    alpaka::ignore_unused(sharedMemKernel);
                    return blockThreadExtent.prod() * threadElemExtent.prod() * static_cast<idx::Idx<TAcc>>(sizeof(Val));
                }
            };
        }
    }
}

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<1u>,
    std::uint32_t>;

TEMPLATE_LIST_TEST_CASE( "sharedMem", "[sharedMem]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    Idx const numElements = 1u<<16u;

    using Val = std::int32_t;
    using TnumUselessWork = std::integral_constant<Idx, 100>;

    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;


    // Create the kernel function object.
    SharedMemKernel<TnumUselessWork, Val> kernel;

    // Select a device to execute on.
    auto const devAcc(
        alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    // Get a queue on this device.
    QueueAcc queue(
        devAcc);

    // Set the grid blocks extent.
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            numElements,
            static_cast<Idx>(1u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "SharedMemKernel("
        << " accelerator: " << alpaka::acc::getAccName<Acc>()
        << ", kernel: " << typeid(kernel).name()
        << ", workDiv: " << workDiv
        << ")" << std::endl;

    Idx const gridBlocksCount(
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[0u]);
    Idx const blockThreadCount(
        alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[0u]);

    // An array for the return values calculated by the blocks.
    std::vector<Val> blockRetVals(static_cast<std::size_t>(gridBlocksCount));

    // Allocate accelerator buffers and copy.
    Idx const resultElemCount(gridBlocksCount);
    auto blockRetValsAcc(alpaka::mem::buf::alloc<Val, Idx>(devAcc, resultElemCount));
    alpaka::mem::view::copy(queue, blockRetValsAcc, blockRetVals, resultElemCount);

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(blockRetValsAcc)));

    // Profile the kernel execution.
    std::cout << "Execution time: "
        << alpaka::test::integ::measureTaskRunTimeMs(
            queue,
            taskKernel)
        << " ms"
        << std::endl;

    // Copy back the result.
    alpaka::mem::view::copy(queue, blockRetVals, blockRetValsAcc, resultElemCount);

    // Wait for the queue to finish the memory operation.
    alpaka::wait::wait(queue);

    // Assert that the results are correct.
    Val const correctResult(
        static_cast<Val>(blockThreadCount*blockThreadCount));

    bool resultCorrect(true);
    for(Idx i(0); i<gridBlocksCount; ++i)
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
