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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <boost/core/ignore_unused.hpp>

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
        //! The CPU OpenMP 2.0 thread accelerator executor.
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp2Threads final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp2Threads(
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
            ExecCpuOmp2Threads(ExecCpuOmp2Threads const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuOmp2Threads(ExecCpuOmp2Threads &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp2Threads const &) -> ExecCpuOmp2Threads & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuOmp2Threads &&) -> ExecCpuOmp2Threads & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuOmp2Threads() = default;

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
                                    acc::AccCpuOmp2Threads<TDim, TSize>>(
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

                acc::AccCpuOmp2Threads<TDim, TSize> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this),
                    blockSharedMemDynSizeBytes);

                // The number of threads in this block.
                TSize const blockThreadCount(blockThreadExtent.prod());
                int const iBlockThreadCount(static_cast<int>(blockThreadCount));
                boost::ignore_unused(iBlockThreadCount);
                // Force the environment to use the given number of threads.
                int const ompIsDynamic(::omp_get_dynamic());
                ::omp_set_dynamic(0);

                // Execute the blocks serially.
                meta::ndLoopIncIdx(
                    gridBlockExtent,
                    [&](vec::Vec<TDim, TSize> const & gridBlockIdx)
                    {
                        acc.m_gridBlockIdx = gridBlockIdx;

                        // Execute the threads in parallel.

                        // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                        // So we have to spawn one OS thread per thread in a block.
                        // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                        // Therefore we use 'omp parallel' with the specified number of threads in a block.
                        #pragma omp parallel num_threads(iBlockThreadCount)
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            // GCC 5.1 fails with:
                            // error: redeclaration of const int& iBlockThreadCount
                            // if(numThreads != iBlockThreadCount
                            //                ^
                            // note: const int& iBlockThreadCount previously declared here
                            // #pragma omp parallel num_threads(iBlockThreadCount)
                            //         ^
#if (!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(5, 0, 0))
                            // The first thread does some checks in the first block executed.
                            if((::omp_get_thread_num() == 0) && (acc.m_gridBlockIdx.sum() == 0u))
                            {
                                int const numThreads(::omp_get_num_threads());
                                std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << numThreads << std::endl;
                                if(numThreads != iBlockThreadCount)
                                {
                                    throw std::runtime_error("The OpenMP 2.0 runtime did not use the number of threads that had been required!");
                                }
                            }
#endif
#endif
                            boundKernelFnObj(
                                acc);

                            // Wait for all threads to finish before deleting the shared memory.
                            // This is done by default if the omp 'nowait' clause is missing
                            //block::sync::syncBlockThreads(acc);
                        }

                        // After a block has been processed, the shared memory has to be deleted.
                        block::shared::st::freeMem(acc);
                    });

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
            //! The CPU OpenMP 2.0 block thread executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp2Threads<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor device type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 2.0 block thread executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 2.0 block thread executor executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block thread executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP 2.0 block thread executor size type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
