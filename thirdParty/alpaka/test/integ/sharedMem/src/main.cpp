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

#include <alpaka/alpaka.hpp>                        // alpaka::exec::create
#include <alpaka/test/MeasureKernelRunTime.hpp>     // measureKernelRunTimeMs
#include <alpaka/test/acc/Acc.hpp>                  // EnabledAccs
#include <alpaka/test/stream/Stream.hpp>            // DefaultStream

#include <iostream>                                 // std::cout
#include <typeinfo>                                 // typeid
#include <vector>                                   // std::vector

//#############################################################################
//! A kernel using atomicOp, syncBlockThreads, getMem, getIdx, getWorkDiv and global memory to compute a (useless) result.
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TnumUselessWork The number of useless calculations done in each kernel execution.
//#############################################################################
template<
    typename TnumUselessWork>
class SharedMemKernel
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    SharedMemKernel(
        std::uint32_t const mult = 2) :
        m_mult(mult)
    {}

    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        std::uint32_t * const puiBlockRetVals,
        std::uint32_t const mult2) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        std::size_t const blockThreadCount(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // Get the dynamically allocated shared memory.
        std::uint32_t * const pBlockShared(alpaka::block::shared::dyn::getMem<std::uint32_t>(acc));

        // Calculate linearized index of the thread in the block.
        std::size_t const blockThreadIdx1d(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        std::uint32_t iSum1(static_cast<std::uint32_t>(blockThreadIdx1d+1));
        for(std::uint32_t i(0); i<TnumUselessWork::value; ++i)
        {
            iSum1 += i;
        }
        pBlockShared[blockThreadIdx1d] = iSum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        alpaka::block::sync::syncBlockThreads(acc);

        // Do something useless.
        std::uint32_t iSum2(static_cast<std::uint32_t>(blockThreadIdx1d));
        for(std::uint32_t i(0); i<TnumUselessWork::value; ++i)
        {
            iSum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        pBlockShared[(blockThreadCount-1)-blockThreadIdx1d] += iSum2;


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
            std::size_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

            puiBlockRetVals[gridBlockIdx] = pBlockShared[0] * m_mult * mult2;
        }
    }

public:
    std::uint32_t const m_mult;
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            //#############################################################################
            template<
                typename TnumUselessWork,
                typename TAcc>
            struct BlockSharedMemDynSizeBytes<
                SharedMemKernel<TnumUselessWork>,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                //-----------------------------------------------------------------------------
                template<
                    typename TVec,
                    typename... TArgs>
                ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                    SharedMemKernel<TnumUselessWork> const & sharedMemKernel,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    TArgs && ...)
                -> std::uint32_t
                {
                    boost::ignore_unused(sharedMemKernel);
                    return blockThreadExtent.prod() * threadElemExtent.prod() * sizeof(std::uint32_t);
                }
            };
        }
    }
}

//#############################################################################
//! Profiles the example kernel and checks the result.
//#############################################################################
template<
    typename TnumUselessWork>
struct SharedMemTester
{
    template<
        typename TAcc,
        typename TSize,
        typename TVal>
    auto operator()(
        TSize const numElements,
        TVal const mult2)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;

        // Create the kernel function object.
        SharedMemKernel<TnumUselessWork> kernel(42);

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0u));

        // Get a stream on this device.
        StreamAcc stream(
            devAcc);

        // Set the grid blocks extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                numElements,
                static_cast<TSize>(1u),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

        std::cout
            << "SharedMemTester("
            << " accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        TSize const gridBlocksCount(
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[0u]);
        TSize const blockThreadCount(
            alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[0u]);

        // An array for the return values calculated by the blocks.
        std::vector<TVal> blockRetVals(gridBlocksCount, static_cast<TVal>(0));

        // Allocate accelerator buffers and copy.
        TSize const resultElemCount(gridBlocksCount);
        auto blockRetValsAcc(alpaka::mem::buf::alloc<TVal, TSize>(devAcc, resultElemCount));
        alpaka::mem::view::copy(stream, blockRetValsAcc, blockRetVals, resultElemCount);

        // Create the executor task.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            alpaka::mem::view::getPtrNative(blockRetValsAcc),
            mult2));

        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::test::integ::measureKernelRunTimeMs(
                stream,
                exec)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(stream, blockRetVals, blockRetValsAcc, resultElemCount);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Assert that the results are correct.
        TVal const correctResult(
            static_cast<TVal>(blockThreadCount*blockThreadCount)
            * kernel.m_mult
            * mult2);

        bool resultCorrect(true);
        for(std::size_t i(0); i<gridBlocksCount; ++i)
        {
            if(blockRetVals[i] != correctResult)
            {
                std::cout << "blockRetVals[" << i << "] == " << blockRetVals[i] << " != " << correctResult << std::endl;
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

public:
    bool allResultsCorrect = true;
};

//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
auto main()
-> int
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka sharedMem test                               " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::test::acc::writeEnabledAccs<alpaka::dim::DimInt<1u>, std::uint32_t>(std::cout);

        std::cout << std::endl;

        using TnumUselessWork = std::integral_constant<std::size_t, 100u>;
        std::uint32_t const mult2(5u);

        SharedMemTester<TnumUselessWork> sharedMemTester;

        // Execute the kernel on all enabled accelerators.
        alpaka::meta::forEachType<
            alpaka::test::acc::EnabledAccs<alpaka::dim::DimInt<1u>, std::uint32_t>>(
                sharedMemTester,
                512u,
                mult2);

        return sharedMemTester.allResultsCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
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
