/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera, Erik Zenker
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

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <boost/assert.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_group.h>

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU TBB block accelerator executor.
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuTbbBlocks final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuTbbBlocks(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            ExecCpuTbbBlocks(ExecCpuTbbBlocks const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuTbbBlocks(ExecCpuTbbBlocks &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuTbbBlocks const &) -> ExecCpuTbbBlocks & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuTbbBlocks &&) -> ExecCpuTbbBlocks & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuTbbBlocks() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(*this));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(*this));

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(
                    meta::apply(
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuTbbBlocks<TDim, TSize>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                // Bind all arguments except the accelerator.
                // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
                auto const boundKernelFnObj(
                    meta::apply(
                        [this](TArgs const & ... args)
                        {
                            return
                                std::bind(
                                    std::ref(m_kernelFnObj),
                                    std::placeholders::_1,
                                    std::ref(args)...);
                        },
                        m_args));

                // The number of blocks in the grid.
                TSize const numBlocksInGrid(gridBlockExtent.prod());

                // There is only ever one thread in a block in the TBB accelerator.
                BOOST_VERIFY(blockThreadExtent.prod() == static_cast<TSize>(1u));

                tbb::parallel_for(
                    static_cast<TSize>(0),
                    static_cast<TSize>(numBlocksInGrid),
                    [&](TSize i){
                         acc::AccCpuTbbBlocks<TDim, TSize> acc(
                             *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this),
                             blockSharedMemDynSizeBytes);

                         acc.m_gridBlockIdx =
                             idx::mapIdx<TDim::value>(
                                 vec::Vec<dim::DimInt<1u>, TSize>(
                                     static_cast<TSize>(i)
                                  ),
                                  gridBlockExtent
                             );

                         boundKernelFnObj(acc);

                         block::shared::st::freeMem(acc);
                });

            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuTbbBlocks<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor device type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU TBB block executor size type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuTbbBlocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
