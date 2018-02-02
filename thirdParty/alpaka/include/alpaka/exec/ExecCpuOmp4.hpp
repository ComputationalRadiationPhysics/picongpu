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

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOmp4.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

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
        //! The CPU OpenMP 4.0 accelerator executor.
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp4 final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp4(
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
            ExecCpuOmp4(ExecCpuOmp4 const & other) = default;
            //-----------------------------------------------------------------------------
            ExecCpuOmp4(ExecCpuOmp4 && other) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp4 const &) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp4 &&) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuOmp4() = default;

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
                                    acc::AccCpuOmp4<TDim, TSize>>(
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
                TSize const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TSize const blockThreadCount(blockThreadExtent.prod());

                // Force the environment to use the given number of threads.
                int const ompIsDynamic(::omp_get_dynamic());
                ::omp_set_dynamic(0);

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                #pragma omp target if(0)
                {
                    #pragma omp teams num_teams(gridBlockCount) thread_limit(blockThreadCount)
                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // The first team does some checks ...
                        if((::omp_get_team_num() == 0))
                        {
                            int const iNumTeams(::omp_get_num_teams());
                            // NOTE: No std::cout in omp target!
                            printf("%s omp_get_num_teams: %d\n", BOOST_CURRENT_FUNCTION, iNumTeams);
                            if(iNumTeams <= 0)    // NOTE: No throw inside target region
                            {
                                throw std::runtime_error("ERROR: The OpenMP runtime did not use a valid number of teams!");
                            }
                        }
#endif
                        acc::AccCpuOmp4<TDim, TSize> acc(
                            *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this),
                            blockSharedMemDynSizeBytes);

                        #pragma omp distribute
                        for(TSize b = 0u; b<gridBlockCount; ++b)
                        {
                            vec::Vec<dim::DimInt<1u>, TSize> const gridBlockIdx(b);
                            // When this is not repeated here:
                            // error: gridBlockExtent referenced in target region does not have a mappable type
                            auto const gridBlockExtent2(
                                workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this)));
                            acc.m_gridBlockIdx = idx::mapIdx<TDim::value>(
                                gridBlockIdx,
                                gridBlockExtent2);

                            // Execute the threads in parallel.

                            // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                            // So we have to spawn one OS thread per thread in a block.
                            // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                            // Therefore we use 'omp parallel' with the specified number of threads in a block.
                            #pragma omp parallel num_threads(blockThreadCount)
                            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                // The first thread does some checks in the first block executed.
                                if((::omp_get_thread_num() == 0) && (b == 0))
                                {
                                    int const numThreads(::omp_get_num_threads());
                                    // NOTE: No std::cout in omp target!
                                    printf("%s omp_get_num_threads: %d\n", BOOST_CURRENT_FUNCTION, numThreads);
                                    if(numThreads != static_cast<int>(blockThreadCount))
                                    {
                                        throw std::runtime_error("ERROR: The OpenMP runtime did not use the number of threads that had been required!");
                                    }
                                }
#endif
                                boundKernelFnObj(
                                    acc);

                                // Wait for all threads to finish before deleting the shared memory.
                                // This is done by default if the omp 'nowait' clause is missing
                                //block::sync::syncBlockThreads(acc);
                            }

                            // After a block has been processed, the shared memory has to be deleted.
                            block::shared::st::freeMem(acc);
                        }
                    }
                }

                // Reset the dynamic thread number setting.
                ::omp_set_dynamic(ompIsDynamic);
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
            //! The CPU OpenMP 4.0 executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp4<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 executor device type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 executor executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 executor size type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
