/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOmp4.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/ApplyTuple.hpp>

#include <omp.h>

#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuOmp4 final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelCpuOmp4(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelCpuOmp4(TaskKernelCpuOmp4 const & other) = default;
            //-----------------------------------------------------------------------------
            TaskKernelCpuOmp4(TaskKernelCpuOmp4 && other) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOmp4 const &) -> TaskKernelCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOmp4 &&) -> TaskKernelCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelCpuOmp4() = default;

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
                        [&](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuOmp4<TDim, TIdx>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                // Bind all arguments except the accelerator.
                // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
                auto const boundKernelFnObj(
                    meta::apply(
                        [this](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                std::bind(
                                    std::ref(m_kernelFnObj),
                                    std::placeholders::_1,
                                    std::ref(args)...);
                        },
                        m_args));

                // The number of blocks in the grid.
                TIdx const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TIdx const blockThreadCount(blockThreadExtent.prod());

                // We have to make sure, that the OpenMP runtime keeps enough threads for executing a block in parallel.
                auto const maxOmpThreadCount(::omp_get_max_threads());
                auto const maxTeamCount(maxOmpThreadCount/static_cast<int>(blockThreadCount));
                auto const teamCount(std::min(maxTeamCount, static_cast<int>(gridBlockCount)));

                if(::omp_in_parallel() != 0)
                {
                    throw std::runtime_error("The OpenMP 4.0 backend can not be used within an existing parallel region!");
                }

                // Force the environment to use the given number of threads.
                int const ompIsDynamic(::omp_get_dynamic());
                ::omp_set_dynamic(0);

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                #pragma omp target if(0)
                {
                    #pragma omp teams num_teams(teamCount) thread_limit(blockThreadCount)
                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // The first team does some checks ...
                        if((::omp_get_team_num() == 0))
                        {
                            int const iNumTeams(::omp_get_num_teams());
                            printf("%s omp_get_num_teams: %d\n", __func__, iNumTeams);
                        }
#endif
                        acc::AccCpuOmp4<TDim, TIdx> acc(
                            *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                            blockSharedMemDynSizeBytes);

                        #pragma omp distribute
                        for(TIdx b = 0u; b<gridBlockCount; ++b)
                        {
                            vec::Vec<dim::DimInt<1u>, TIdx> const gridBlockIdx(b);
                            // When this is not repeated here:
                            // error: gridBlockExtent referenced in target region does not have a mappable type
                            auto const gridBlockExtent2(
                                workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this)));
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
                                    printf("%s omp_get_num_threads: %d\n", __func__, numThreads);
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

        private:
            TKernelFnObj m_kernelFnObj;
            std::tuple<std::decay_t<TArgs>...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp4<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 4.0 execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
