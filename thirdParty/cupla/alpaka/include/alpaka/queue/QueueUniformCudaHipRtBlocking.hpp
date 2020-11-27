/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/queue/cuda_hip/QueueUniformCudaHipRtBase.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <condition_variable>
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <stdexcept>
#    include <thread>

namespace alpaka
{
    class EventUniformCudaHipRt;

    //#############################################################################
    //! The CUDA/HIP RT blocking queue.
    class QueueUniformCudaHipRtBlocking final : public uniform_cuda_hip::detail::QueueUniformCudaHipRtBase
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueUniformCudaHipRtBlocking(DevUniformCudaHipRt const& dev)
            : uniform_cuda_hip::detail::QueueUniformCudaHipRtBase(dev)
        {
        }
        //-----------------------------------------------------------------------------
        QueueUniformCudaHipRtBlocking(QueueUniformCudaHipRtBlocking const&) = default;
        //-----------------------------------------------------------------------------
        QueueUniformCudaHipRtBlocking(QueueUniformCudaHipRtBlocking&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueUniformCudaHipRtBlocking const&) -> QueueUniformCudaHipRtBlocking& = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueUniformCudaHipRtBlocking&&) -> QueueUniformCudaHipRtBlocking& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRtBlocking const& rhs) const -> bool
        {
            return (m_spQueueImpl == rhs.m_spQueueImpl);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRtBlocking const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueUniformCudaHipRtBlocking() = default;
    };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using QueueCudaRtBlocking = QueueUniformCudaHipRtBlocking;
#    else
    using QueueHipRtBlocking = QueueUniformCudaHipRtBlocking;
#    endif

    namespace traits
    {
        //#############################################################################
        //! The CUDA/HIP RT blocking queue device type trait specialization.
        template<>
        struct DevType<QueueUniformCudaHipRtBlocking>
        {
            using type = DevUniformCudaHipRt;
        };

        //#############################################################################
        //! The CUDA/HIP RT blocking queue event type trait specialization.
        template<>
        struct EventType<QueueUniformCudaHipRtBlocking>
        {
            using type = EventUniformCudaHipRt;
        };

        //#############################################################################
        //! The CUDA/HIP RT blocking queue enqueue trait specialization.
        template<typename TTask>
        struct Enqueue<QueueUniformCudaHipRtBlocking, TTask>
        {
            //#############################################################################
            enum class CallbackState
            {
                enqueued,
                notified,
                finished,
            };

            //#############################################################################
            struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
            {
                std::mutex m_mutex;
                std::condition_variable m_event;
                CallbackState state = CallbackState::enqueued;
            };

            //-----------------------------------------------------------------------------
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            static void CUDART_CB
#    else
            static void HIPRT_CB
#    endif
            uniformCudaHipRtCallback(
                ALPAKA_API_PREFIX(Stream_t) /*queue*/,
                ALPAKA_API_PREFIX(Error_t) /*status*/,
                void* arg)
            {
                // explicitly copy the shared_ptr so that this method holds the state even when the executing thread
                // has already finished.
                const auto pCallbackSynchronizationData
                    = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                // Notify the executing thread.
                {
                    std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                    pCallbackSynchronizationData->state = CallbackState::notified;
                }
                pCallbackSynchronizationData->m_event.notify_one();

                // Wait for the executing thread to finish the task if it has not already finished.
                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                if(pCallbackSynchronizationData->state != CallbackState::finished)
                {
                    pCallbackSynchronizationData->m_event.wait(lock, [pCallbackSynchronizationData]() {
                        return pCallbackSynchronizationData->state == CallbackState::finished;
                    });
                }
            }

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtBlocking& queue, TTask const& task) -> void
            {
                auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamAddCallback)(
                    queue.m_spQueueImpl->m_UniformCudaHipQueue,
                    uniformCudaHipRtCallback,
                    pCallbackSynchronizationData.get(),
                    0u));

                // We start a new std::thread which stores the task to be executed.
                // This circumvents the limitation that it is not possible to call CUDA/HIP methods within the CUDA/HIP
                // callback thread. The CUDA/HIP thread signals the std::thread when it is ready to execute the task.
                // The CUDA/HIP thread is waiting for the std::thread to signal that it is finished executing the task
                // before it executes the next task in the queue (CUDA/HIP stream).
                std::thread t([pCallbackSynchronizationData, task]() {
                    // If the callback has not yet been called, we wait for it.
                    {
                        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                        if(pCallbackSynchronizationData->state != CallbackState::notified)
                        {
                            pCallbackSynchronizationData->m_event.wait(lock, [pCallbackSynchronizationData]() {
                                return pCallbackSynchronizationData->state == CallbackState::notified;
                            });
                        }

                        task();

                        // Notify the waiting CUDA/HIP thread.
                        pCallbackSynchronizationData->state = CallbackState::finished;
                    }
                    pCallbackSynchronizationData->m_event.notify_one();
                });

                t.join();
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
