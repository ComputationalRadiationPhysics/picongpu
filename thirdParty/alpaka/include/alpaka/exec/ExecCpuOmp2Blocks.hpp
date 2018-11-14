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

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/ApplyTuple.hpp>

#include <omp.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator executor.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp2Blocks final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {

                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            ExecCpuOmp2Blocks(ExecCpuOmp2Blocks const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuOmp2Blocks(ExecCpuOmp2Blocks &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp2Blocks const &) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp2Blocks &&) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuOmp2Blocks() = default;

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
                                    acc::AccCpuOmp2Blocks<TDim, TIdx>>(
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
                TIdx const numBlocksInGrid(gridBlockExtent.prod());
                if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error("Only one thread per block allowed in the OpenMP 2.0 block accelerator!");
                }

                if(::omp_in_parallel() != 0)
                {
                    parallelFn(
                        boundKernelFnObj,
                        blockSharedMemDynSizeBytes,
                        numBlocksInGrid,
                        gridBlockExtent);
                }
                else
                {
                    #pragma omp parallel
                    parallelFn(
                        boundKernelFnObj,
                        blockSharedMemDynSizeBytes,
                        numBlocksInGrid,
                        gridBlockExtent);
                }
            }

        private:
            template<
                typename FnObj>
            ALPAKA_FN_HOST auto parallelFn(
                FnObj const & boundKernelFnObj,
                TIdx const & blockSharedMemDynSizeBytes,
                TIdx const & numBlocksInGrid,
                vec::Vec<TDim, TIdx> const & gridBlockExtent) const
            -> void
            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // The first thread does some debug logging.
                if(::omp_get_thread_num() == 0)
                {
                    int const numThreads(::omp_get_num_threads());
                    std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << numThreads << std::endl;
                }
#endif
                acc::AccCpuOmp2Blocks<TDim, TIdx> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                    blockSharedMemDynSizeBytes);

                // NOTE: schedule(static) does not improve performance.
#if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop header.
                std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numBlocksInGrid));
                std::intmax_t i;
                #pragma omp for nowait schedule(guided)
                for(i = 0; i < iNumBlocksInGrid; ++i)
#else
                #pragma omp for nowait schedule(guided)
                for(TIdx i = 0; i < numBlocksInGrid; ++i)
#endif
                {
                    acc.m_gridBlockIdx =
                        idx::mapIdx<TDim::value>(
#if _OPENMP < 200805
                            vec::Vec<dim::DimInt<1u>, TIdx>(static_cast<TIdx>(i)),
#else
                            vec::Vec<dim::DimInt<1u>, TIdx>(i),
#endif
                            gridBlockExtent);

                    boundKernelFnObj(
                        acc);

                    // After a block has been processed, the shared memory has to be deleted.
                    block::shared::st::freeMem(acc);
                }
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
            //! The CPU OpenMP 2.0 grid block executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp2Blocks<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 2.0 grid block executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block executor idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                exec::ExecCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
