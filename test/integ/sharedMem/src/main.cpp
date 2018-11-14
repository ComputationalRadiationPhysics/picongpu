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

#define BOOST_TEST_MODULE sharedMem

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>

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

BOOST_AUTO_TEST_SUITE(sharedMem)

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<1u>,
    std::uint32_t>;

BOOST_AUTO_TEST_CASE_TEMPLATE(
    calculateAxpy,
    TAcc,
    TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    Idx const numElements = 1u<<16u;

    using Val = std::int32_t;
    using TnumUselessWork = std::integral_constant<Idx, 100>;

    using DevAcc = alpaka::dev::Dev<TAcc>;
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
        alpaka::workdiv::getValidWorkDiv<TAcc>(
            devAcc,
            numElements,
            static_cast<Idx>(1u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "SharedMemKernel("
        << " accelerator: " << alpaka::acc::getAccName<TAcc>()
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

    // Create the executor task.
    auto const exec(alpaka::kernel::createTaskExec<TAcc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(blockRetValsAcc)));

    // Profile the kernel execution.
    std::cout << "Execution time: "
        << alpaka::test::integ::measureTaskRunTimeMs(
            queue,
            exec)
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

    BOOST_REQUIRE_EQUAL(true, resultCorrect);
}

BOOST_AUTO_TEST_SUITE_END()
