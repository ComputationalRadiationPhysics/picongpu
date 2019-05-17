/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test queue specifics.
        namespace queue
        {
            namespace traits
            {
                //#############################################################################
                //! The default queue type trait for devices.
                template<
                    typename TDev,
                    typename TSfinae = void>
                struct DefaultQueueType;

                //#############################################################################
                //! The default queue type trait specialization for the CPU device.
                template<>
                struct DefaultQueueType<
                    alpaka::dev::DevCpu>
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::queue::QueueCpuSync;
#else
                    using type = alpaka::queue::QueueCpuAsync;
#endif
                };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif
                //#############################################################################
                //! The default queue type trait specialization for the CUDA device.
                template<>
                struct DefaultQueueType<
                    alpaka::dev::DevCudaRt>
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::queue::QueueCudaRtSync;
#else
                    using type = alpaka::queue::QueueCudaRtAsync;
#endif
                };
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif
                //#############################################################################
                //! The default queue type trait specialization for the HIP device.
                template<>
                struct DefaultQueueType<
                    alpaka::dev::DevHipRt>
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::queue::QueueHipRtSync;
#else
                    using type = alpaka::queue::QueueHipRtAsync;
#endif
                };
#endif

            }
            //#############################################################################
            //! The queue type that should be used for the given accelerator.
            template<
                typename TAcc>
            using DefaultQueue = typename traits::DefaultQueueType<TAcc>::type;

            namespace traits
            {
                //#############################################################################
                //! The sync queue trait.
                template<
                    typename TQueue,
                    typename TSfinae = void>
                struct IsSyncQueue;

                //#############################################################################
                //! The sync queue trait specialization for a sync CPU queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueCpuSync>
                {
                    static constexpr bool value = true;
                };

                //#############################################################################
                //! The sync queue trait specialization for a async CPU queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueCpuAsync>
                {
                    static constexpr bool value = false;
                };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif
                //#############################################################################
                //! The sync queue trait specialization for a sync CUDA RT queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueCudaRtSync>
                {
                    static constexpr bool value = true;
                };

                //#############################################################################
                //! The sync queue trait specialization for a async CUDA RT queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueCudaRtAsync>
                {
                    static constexpr bool value = false;
                };
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif
                //#############################################################################
                //! The sync queue trait specialization for a sync HIP RT queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueHipRtSync>
                {
                    static constexpr bool value = true;
                };

                //#############################################################################
                //! The sync queue trait specialization for a async HIP RT queue.
                template<>
                struct IsSyncQueue<
                    alpaka::queue::QueueHipRtAsync>
                {
                    static constexpr bool value = false;
                };
#endif
            }
            //#############################################################################
            //! The queue type that should be used for the given accelerator.
            template<
                typename TQueue>
            using IsSyncQueue = traits::IsSyncQueue<TQueue>;

            //#############################################################################
            //! A std::tuple holding tuples of devices and corresponding queue types.
            using TestQueues =
                std::tuple<
                    std::tuple<alpaka::dev::DevCpu, alpaka::queue::QueueCpuSync>,
                    std::tuple<alpaka::dev::DevCpu, alpaka::queue::QueueCpuAsync>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    ,
                    std::tuple<alpaka::dev::DevCudaRt, alpaka::queue::QueueCudaRtSync>,
                    std::tuple<alpaka::dev::DevCudaRt, alpaka::queue::QueueCudaRtAsync>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
                    ,
                    std::tuple<alpaka::dev::DevHipRt, alpaka::queue::QueueHipRtSync>,
                    std::tuple<alpaka::dev::DevHipRt, alpaka::queue::QueueHipRtAsync>
#endif
                >;
        }
    }
}
