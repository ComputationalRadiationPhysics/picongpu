/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

#pragma once

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Debug.hpp>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    #include <alpaka/workdiv/Traits.hpp>
#endif

#include <type_traits>

//-----------------------------------------------------------------------------
//! The alpaka accelerator library.
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The kernel specifics.
    namespace kernel
    {
        //-----------------------------------------------------------------------------
        //! The kernel traits.
        namespace traits
        {
            //#############################################################################
            //! The kernel execution task creation trait.
            template<
                typename TAcc,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs/*,
                typename TSfinae = void*/>
            struct CreateTaskExec;

            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory of a kernel.
            //!
            //! \tparam TKernelFnObj The kernel function object.
            //! \tparam TAcc The accelerator.
            //!
            //! The default implementation returns 0.
            template<
                typename TKernelFnObj,
                typename TAcc,
                typename TSfinae = void>
            struct BlockSharedMemDynSizeBytes
            {
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdocumentation"  // clang does not support the syntax for variadic template arguments "args,..."
#endif
                //-----------------------------------------------------------------------------
                //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
                //! \param blockThreadExtent The block thread extent.
                //! \param threadElemExtent The thread element extent.
                //! \tparam TArgs The kernel invocation argument types pack.
                //! \param args,... The kernel invocation arguments.
                //! \return The size of the shared memory allocated for a block in bytes.
                //! The default version always returns zero.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TDim,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    TKernelFnObj const & kernelFnObj,
                    vec::Vec<TDim, idx::Idx<TAcc>> const & blockThreadExtent,
                    vec::Vec<TDim, idx::Idx<TAcc>> const & threadElemExtent,
                    TArgs const & ... args)
                -> idx::Idx<TAcc>
                {
                    alpaka::ignore_unused(kernelFnObj);
                    alpaka::ignore_unused(blockThreadExtent);
                    alpaka::ignore_unused(threadElemExtent);
                    alpaka::ignore_unused(args...);

                    return 0;
                }
            };
        }

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdocumentation"  // clang does not support the syntax for variadic template arguments "args,..."
#endif
        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator type.
        //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
        //! \param blockThreadExtent The block thread extent.
        //! \param threadElemExtent The thread element extent.
        //! \param args,... The kernel invocation arguments.
        //! \return The size of the shared memory allocated for a block in bytes.
        //! The default implementation always returns zero.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TKernelFnObj,
            typename TDim,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto getBlockSharedMemDynSizeBytes(
            TKernelFnObj const & kernelFnObj,
            vec::Vec<TDim, idx::Idx<TAcc>> const & blockThreadExtent,
            vec::Vec<TDim, idx::Idx<TAcc>> const & threadElemExtent,
            TArgs const & ... args)
        -> idx::Idx<TAcc>
        {
            return
                traits::BlockSharedMemDynSizeBytes<
                    TKernelFnObj,
                    TAcc>
                ::getBlockSharedMemDynSizeBytes(
                    kernelFnObj,
                    blockThreadExtent,
                    threadElemExtent,
                    args...);
        }

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdocumentation"  // clang does not support the syntax for variadic template arguments "args,..."
#endif
        //-----------------------------------------------------------------------------
        //! Creates a kernel execution task.
        //!
        //! \tparam TAcc The accelerator type.
        //! \param workDiv The index domain work division.
        //! \param kernelFnObj The kernel function object which should be executed.
        //! \param args,... The kernel invocation arguments.
        //! \return The kernel execution task.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
        template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST auto createTaskExec(
            TWorkDiv const & workDiv,
            TKernelFnObj const & kernelFnObj,
            TArgs const & ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::CreateTaskExec<
                TAcc,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            ::createTaskExec(
                workDiv,
                kernelFnObj,
                args...))
#endif
        {
            static_assert(
                dim::Dim<typename std::decay<TWorkDiv>::type>::value == dim::Dim<TAcc>::value,
                "The dimensions of TAcc and TWorkDiv have to be identical!");
            static_assert(
                std::is_same<idx::Idx<typename std::decay<TWorkDiv>::type>, idx::Idx<TAcc>>::value,
                "The idx type of TAcc and the idx type of TWorkDiv have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << BOOST_CURRENT_FUNCTION
                << " gridBlockExtent: " << workdiv::getWorkDiv<Grid, Blocks>(workDiv)
                << ", blockThreadExtent: " << workdiv::getWorkDiv<Block, Threads>(workDiv)
                << std::endl;
#endif
            return
                traits::CreateTaskExec<
                    TAcc,
                    TWorkDiv,
                    TKernelFnObj,
                    TArgs...>::createTaskExec(
                        workDiv,
                        kernelFnObj,
                        args...);
        }

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdocumentation"  // clang does not support the syntax for variadic template arguments "args,..."
#endif
        //-----------------------------------------------------------------------------
        //! Executes the given kernel in the given queue.
        //!
        //! \tparam TAcc The accelerator type.
        //! \param queue The queue to enqueue the view copy task into.
        //! \param workDiv The index domain work division.
        //! \param kernelFnObj The kernel function object which should be executed.
        //! \param args,... The kernel invocation arguments.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
        template<
            typename TAcc,
            typename TQueue,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST auto exec(
            TQueue & queue,
            TWorkDiv const & workDiv,
            TKernelFnObj const & kernelFnObj,
            TArgs const & ... args)
        -> void
        {
            queue::enqueue(
                queue,
                kernel::createTaskExec<
                    TAcc>(
                    workDiv,
                    kernelFnObj,
                    args...));
        }
    }
}
