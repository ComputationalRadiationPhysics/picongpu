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

#include <boost/predef.h>
#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

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
#if !BOOST_ARCH_CUDA_DEVICE
                    TKernelFnObj const & kernelFnObj,
                    vec::Vec<TDim, size::Size<TAcc>> const & blockThreadExtent,
                    vec::Vec<TDim, size::Size<TAcc>> const & threadElemExtent,
                    TArgs const & ... args)
#else
                    TKernelFnObj const &,
                    vec::Vec<TDim, size::Size<TAcc>> const &,
                    vec::Vec<TDim, size::Size<TAcc>> const &,
                    TArgs const & ...)
#endif
                -> size::Size<TAcc>
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(kernelFnObj);
                    boost::ignore_unused(blockThreadExtent);
                    boost::ignore_unused(threadElemExtent);
                    boost::ignore_unused(args...);
#endif

                    return 0;
                }
            };
        }

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
            vec::Vec<TDim, size::Size<TAcc>> const & blockThreadExtent,
            vec::Vec<TDim, size::Size<TAcc>> const & threadElemExtent,
            TArgs const & ... args)
        -> size::Size<TAcc>
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
    }
}
