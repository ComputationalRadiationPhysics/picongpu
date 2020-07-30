/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
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
                    using type = alpaka::queue::QueueCpuBlocking;
#else
                    using type = alpaka::queue::QueueCpuNonBlocking;
#endif
                };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

                //#############################################################################
                //! The default queue type trait specialization for the CUDA/HIP device.
                template<>
                struct DefaultQueueType<
                    alpaka::dev::DevUniformCudaHipRt>
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::queue::QueueUniformCudaHipRtBlocking;
#else
                    using type = alpaka::queue::QueueUniformCudaHipRtNonBlocking;
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
                //! The blocking queue trait.
                template<
                    typename TQueue,
                    typename TSfinae = void>
                struct IsBlockingQueue;

                //#############################################################################
                //! The blocking queue trait specialization for a blocking CPU queue.
                template<>
                struct IsBlockingQueue<
                    alpaka::queue::QueueCpuBlocking>
                {
                    static constexpr bool value = true;
                };

                //#############################################################################
                //! The blocking queue trait specialization for a non-blocking CPU queue.
                template<>
                struct IsBlockingQueue<
                    alpaka::queue::QueueCpuNonBlocking>
                {
                    static constexpr bool value = false;
                };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

                //#############################################################################
                //! The blocking queue trait specialization for a blocking CUDA/HIP RT queue.
                template<>
                struct IsBlockingQueue<
                    alpaka::queue::QueueUniformCudaHipRtBlocking>
                {
                    static constexpr bool value = true;
                };

                //#############################################################################
                //! The blocking queue trait specialization for a non-blocking CUDA/HIP RT queue.
                template<>
                struct IsBlockingQueue<
                    alpaka::queue::QueueUniformCudaHipRtNonBlocking>
                {
                    static constexpr bool value = false;
                };
#endif

            }
            //#############################################################################
            //! The queue type that should be used for the given accelerator.
            template<
                typename TQueue>
            using IsBlockingQueue = traits::IsBlockingQueue<TQueue>;

            //#############################################################################
            //! A std::tuple holding tuples of devices and corresponding queue types.
            using TestQueues =
                std::tuple<
                    std::tuple<alpaka::dev::DevCpu, alpaka::queue::QueueCpuBlocking>,
                    std::tuple<alpaka::dev::DevCpu, alpaka::queue::QueueCpuNonBlocking>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    ,
                    std::tuple<alpaka::dev::DevUniformCudaHipRt, alpaka::queue::QueueUniformCudaHipRtBlocking>,
                    std::tuple<alpaka::dev::DevUniformCudaHipRt, alpaka::queue::QueueUniformCudaHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
                    ,
                    std::tuple<alpaka::dev::DevHipRt, alpaka::queue::QueueHipRtBlocking>,
                    std::tuple<alpaka::dev::DevHipRt, alpaka::queue::QueueHipRtNonBlocking>
#endif
                >;
        }
    }
}
