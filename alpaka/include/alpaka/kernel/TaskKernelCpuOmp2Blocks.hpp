/* Copyright 2019-2020 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/OmpSchedule.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/ApplyTuple.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <omp.h>

#    include <functional>
#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    //#############################################################################
    //! The CPU OpenMP 2.0 block accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuOmp2Blocks final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuOmp2Blocks(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }
        //-----------------------------------------------------------------------------
        TaskKernelCpuOmp2Blocks(TaskKernelCpuOmp2Blocks const&) = default;
        //-----------------------------------------------------------------------------
        TaskKernelCpuOmp2Blocks(TaskKernelCpuOmp2Blocks&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuOmp2Blocks const&) -> TaskKernelCpuOmp2Blocks& = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuOmp2Blocks&&) -> TaskKernelCpuOmp2Blocks& = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelCpuOmp2Blocks() = default;

        //-----------------------------------------------------------------------------
        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(*this));
            auto const blockThreadExtent(getWorkDiv<Block, Threads>(*this));
            auto const threadElemExtent(getWorkDiv<Thread, Elems>(*this));

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes(meta::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return getBlockSharedMemDynSizeBytes<AccCpuOmp2Blocks<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            // Bind all arguments except the accelerator.
            // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
            auto const boundKernelFnObj(meta::apply(
                [this](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return std::bind(std::ref(m_kernelFnObj), std::placeholders::_1, std::ref(args)...);
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
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " already within a parallel region." << std::endl;
#    endif
                parallelFn(boundKernelFnObj, blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent);
            }
            else
            {
                //! Set the given OpenMP schedule while this object is alive.
                //! Restore the old schedule afterwards.
                class ScheduleGuard
                {
                public:
                    ScheduleGuard(omp::Schedule const schedule) : oldSchedule(omp::getSchedule())
                    {
                        omp::setSchedule(schedule);
                    }

                    ScheduleGuard(ScheduleGuard const&) = default;

                    ~ScheduleGuard()
                    {
                        omp::setSchedule(oldSchedule);
                    }

                private:
                    omp::Schedule const oldSchedule;
                };

                // Get the OpenMP schedule.
                // We only do it when outside of a parallel region, since
                // otherwise the change of schedule would have no effect.
                auto const schedule(meta::apply(
                    [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                        return getOmpSchedule<AccCpuOmp2Blocks<TDim, TIdx>>(
                            m_kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
                    },
                    m_args));

                // Schedule change is a scoped object, so that the old schedule is
                // also restored in case of exception.
                auto const scheduleGuard = ScheduleGuard{schedule};

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " opening new parallel region." << std::endl;
#    endif
#    pragma omp parallel
                parallelFn(boundKernelFnObj, blockSharedMemDynSizeBytes, numBlocksInGrid, gridBlockExtent);
            }
        }

    private:
        template<typename FnObj>
        ALPAKA_FN_HOST auto parallelFn(
            FnObj const& boundKernelFnObj,
            std::size_t const& blockSharedMemDynSizeBytes,
            TIdx const& numBlocksInGrid,
            Vec<TDim, TIdx> const& gridBlockExtent) const -> void
        {
#    pragma omp single nowait
            {
                // The OpenMP runtime does not create a parallel region when either:
                // * only one thread is required in the num_threads clause
                // * or only one thread is available
                // In all other cases we expect to be in a parallel region now.
                if((numBlocksInGrid > 1) && (::omp_get_max_threads() > 1) && (::omp_in_parallel() == 0))
                {
                    throw std::runtime_error("The OpenMP 2.0 runtime did not create a parallel region!");
                }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << __func__ << " omp_get_num_threads: " << ::omp_get_num_threads() << std::endl;
#    endif
            }

            AccCpuOmp2Blocks<TDim, TIdx> acc(
                *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                blockSharedMemDynSizeBytes);

#    if _OPENMP < 200805 // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop
                         // header.
            std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numBlocksInGrid));
            std::intmax_t i;
#        pragma omp for nowait schedule(runtime)
            for(i = 0; i < iNumBlocksInGrid; ++i)
#    else
#        pragma omp for nowait schedule(runtime)
            for(TIdx i = 0; i < numBlocksInGrid; ++i)
#    endif
            {
#    if _OPENMP < 200805
                auto const i_tidx = static_cast<TIdx>(i); // for issue #840
                auto const index = Vec<DimInt<1u>, TIdx>(i_tidx); // for issue #840
#    else
                auto const index = Vec<DimInt<1u>, TIdx>(i); // for issue #840
#    endif
                acc.m_gridBlockIdx = mapIdx<TDim::value>(index, gridBlockExtent);

                boundKernelFnObj(acc);

                // After a block has been processed, the shared memory has to be deleted.
                freeSharedVars(acc);
            }
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuOmp2Blocks<TDim, TIdx>;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 grid block execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU OpenMP 2.0 block execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
