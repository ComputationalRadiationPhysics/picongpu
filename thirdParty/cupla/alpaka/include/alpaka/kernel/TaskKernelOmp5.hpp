/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccOmp5.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Tuple.hpp>
#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <omp.h>

#    include <algorithm>
#    include <functional>
#    include <stdexcept>
#    include <type_traits>

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    //! The OpenMP 5.0 accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelOmp5 final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelOmp5(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()(DevOmp5 const& dev) const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cout << "m_gridBlockExtent=" << this->m_gridBlockExtent << "\tgridBlockExtent=" << gridBlockExtent
                      << std::endl;
            std::cout << "m_blockThreadExtent=" << this->m_blockThreadExtent
                      << "\tblockThreadExtent=" << blockThreadExtent << std::endl;
            std::cout << "m_threadElemExtent=" << this->m_threadElemExtent << "\tthreadElemExtent=" << threadElemExtent
                      << std::endl;
#    endif

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes = core::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args)
                {
                    return getBlockSharedMemDynSizeBytes<AccOmp5<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            // We have to make sure, that the OpenMP runtime keeps enough threads for executing a block in parallel.
            TIdx const maxOmpThreadCount(static_cast<TIdx>(::omp_get_max_threads()));
            // The number of blocks in the grid.
            TIdx const gridBlockCount(gridBlockExtent.prod());
            // The number of threads in a block.
            TIdx const blockThreadCount(blockThreadExtent.prod());

            if(gridBlockCount == 0 || blockThreadCount == 0)
            { //! empty grid is a NOP
                return;
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            if(maxOmpThreadCount < blockThreadExtent.prod())
            {
                std::cout
                    << "Warning: TaskKernelOmp5: maxOmpThreadCount smaller than blockThreadCount requested by caller:"
                    << maxOmpThreadCount << " < " << blockThreadExtent.prod() << std::endl;
            }
#    endif
            // make sure there is at least on team
            TIdx const teamCount(std::max(
                std::min(static_cast<TIdx>(maxOmpThreadCount / blockThreadCount), gridBlockCount),
                static_cast<TIdx>(1u)));
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << "threadElemCount=" << threadElemExtent[0u] << std::endl;
            std::cout << "teamCount=" << teamCount << "\tgridBlockCount=" << gridBlockCount << std::endl;
#    endif

            if(::omp_in_parallel() != 0)
            {
                throw std::runtime_error("The OpenMP 5.0 backend can not be used within an existing parallel region!");
            }

            // Force the environment to use the given number of threads.
            int const ompIsDynamic(::omp_get_dynamic());
            ::omp_set_dynamic(0);

            // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
            auto argsD = m_args;
            auto kernelFnObj = m_kernelFnObj;
            auto const iDevice = dev.getNativeHandle();
#    pragma omp target device(iDevice)
            {
#    pragma omp teams distribute num_teams(teamCount) // thread_limit(blockThreadCount)
                for(TIdx t = 0u; t < gridBlockCount; ++t)
                {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL || defined ALPAKA_CI
                    // The first team does some checks ...
                    if(t == 0)
                    {
                        int const iNumTeams(::omp_get_num_teams());
                        printf("%s omp_get_num_teams: %d\n", __func__, iNumTeams);
                        printf("threadElemCount_dev %d\n", int(threadElemExtent[0u]));
                    }
#    endif
                    AccOmp5<TDim, TIdx>
                        acc(gridBlockExtent, blockThreadExtent, threadElemExtent, t, blockSharedMemDynSizeBytes);

                    // Execute the threads in parallel.

                    // Parallel execution of the threads in a block is required because when syncBlockThreads is called
                    // all of them have to be done with their work up to this line. So we have to spawn one OS thread
                    // per thread in a block. 'omp for' is not useful because it is meant for cases where multiple
                    // iterations are executed by one thread but in our case a 1:1 mapping is required. Therefore we
                    // use 'omp parallel' with the specified number of threads in a block.
#    ifndef __ibmxl_vrm__
// setting num_threads to any value leads XL to run only one thread per team
#        pragma omp parallel num_threads(blockThreadCount)
#    else
#        pragma omp parallel
#    endif
                    {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL || defined ALPAKA_CI
                        // The first thread does some checks in the first block executed.
                        if((::omp_get_thread_num() == 0) && (t == 0))
                        {
                            int const numThreads = ::omp_get_num_threads();
                            printf("%s omp_get_num_threads: %d\n", __func__, numThreads);
                            if(numThreads != static_cast<int>(blockThreadCount))
                            {
                                printf("ERROR: The OpenMP runtime did not use the number of threads that had been "
                                       "requested!\n");
                            }
                        }
#    endif
                        core::apply(
                            [kernelFnObj, &acc](typename std::decay<TArgs>::type const&... args)
                            { kernelFnObj(acc, args...); },
                            argsD);

                        // Wait for all threads to finish before deleting the shared memory.
                        // This is done by default if the omp 'nowait' clause is missing
                        // syncBlockThreads(acc);
                    }
                }
            }

            // Reset the dynamic thread number setting.
            ::omp_set_dynamic(ompIsDynamic);
        }

    private:
        TKernelFnObj m_kernelFnObj;
        core::Tuple<std::decay_t<TArgs>...> m_args;
    };
    namespace trait
    {
        //! The OpenMP 5.0 execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccOmp5<TDim, TIdx>;
        };

        //! The OpenMP 5.0 execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevOmp5;
        };

        //! The OpenMP 5.0 execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The OpenMP 5.0 execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfOmp5;
        };

        //! The OpenMP 5.0 execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueOmp5Blocking, TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueOmp5Blocking& queue,
                TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                task(queue.m_spQueueImpl->m_dev);

                queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
            }
        };

        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueOmp5NonBlocking, TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueOmp5NonBlocking& queue,
                TaskKernelOmp5<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                queue.m_spQueueImpl->m_workerThread->enqueueTask([&queue, task]()
                                                                 { task(queue.m_spQueueImpl->m_dev); });
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
