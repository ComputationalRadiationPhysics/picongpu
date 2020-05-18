/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/DevUniformCudaHipRt.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace alpaka
{
    namespace event
    {
        class EventUniformCudaHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace uniform_cuda_hip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA/HIP RT blocking queue implementation.
                class QueueUniformCudaHipRtBlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueUniformCudaHipRtBlockingImpl(
                        dev::DevUniformCudaHipRt const & dev) :
                            m_dev(dev),
                            m_UniformCudaHipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            ALPAKA_API_PREFIX(SetDevice)(
                                m_dev.m_iDevice));

                        // - [cuda/hip]StreamDefault: Default queue creation flag.
                        // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run concurrently with work in queue 0 (the NULL queue),
                        //   and that the created queue should perform no implicit synchronization with queue 0.
                        // Create the queue on the current device.
                        // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka CPU queue.
                        // It would be too much work to implement implicit default queue synchronization on CPU.

                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            ALPAKA_API_PREFIX(StreamCreateWithFlags)(
                                &m_UniformCudaHipQueue,
                                ALPAKA_API_PREFIX(StreamNonBlocking)));

                    }
                    //-----------------------------------------------------------------------------
                    QueueUniformCudaHipRtBlockingImpl(QueueUniformCudaHipRtBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueUniformCudaHipRtBlockingImpl(QueueUniformCudaHipRtBlockingImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueUniformCudaHipRtBlockingImpl const &) -> QueueUniformCudaHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueUniformCudaHipRtBlockingImpl &&) -> QueueUniformCudaHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueUniformCudaHipRtBlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before [cuda/hip]StreamDestroy required?

                        // In case the device is still doing work in the queue when [cuda/hip]StreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.

                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            ALPAKA_API_PREFIX(SetDevice)(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the queue when cuda/hip-StreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            ALPAKA_API_PREFIX(StreamDestroy)(
                                m_UniformCudaHipQueue));

                    }

                public:
                    dev::DevUniformCudaHipRt const m_dev;   //!< The device this queue is bound to.
                    ALPAKA_API_PREFIX(Stream_t) m_UniformCudaHipQueue;
                };
            }
        }

        //#############################################################################
        //! The CUDA/HIP RT blocking queue.
        class QueueUniformCudaHipRtBlocking final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueUniformCudaHipRtBlocking>
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueUniformCudaHipRtBlocking(
                dev::DevUniformCudaHipRt const & dev) :
                m_spQueueImpl(std::make_shared<uniform_cuda_hip::detail::QueueUniformCudaHipRtBlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueUniformCudaHipRtBlocking(QueueUniformCudaHipRtBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueUniformCudaHipRtBlocking(QueueUniformCudaHipRtBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueUniformCudaHipRtBlocking const &) -> QueueUniformCudaHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueUniformCudaHipRtBlocking &&) -> QueueUniformCudaHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRtBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRtBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueUniformCudaHipRtBlocking() = default;

        public:
            std::shared_ptr<uniform_cuda_hip::detail::QueueUniformCudaHipRtBlockingImpl> m_spQueueImpl;
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        using QueueCudaRtBlocking = QueueUniformCudaHipRtBlocking;
#else
        using QueueHipRtBlocking = QueueUniformCudaHipRtBlocking;
#endif
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueUniformCudaHipRtBlocking>
            {
                using type = dev::DevUniformCudaHipRt;
            };
            //#############################################################################
            //! The CUDA/HIP RT blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueUniformCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueUniformCudaHipRtBlocking const & queue)
                -> dev::DevUniformCudaHipRt
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueUniformCudaHipRtBlocking>
            {
                using type = event::EventUniformCudaHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueUniformCudaHipRtBlocking,
                TTask>
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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                static void CUDART_CB
#else
                static void HIPRT_CB
#endif
                uniformCudaHipRtCallback(ALPAKA_API_PREFIX(Stream_t) /*queue*/, ALPAKA_API_PREFIX(Error_t) /*status*/, void *arg)
                {
                    // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
                    const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

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
                        pCallbackSynchronizationData->m_event.wait(
                            lock,
                            [pCallbackSynchronizationData](){
                                return pCallbackSynchronizationData->state == CallbackState::finished;
                            }
                        );
                    }
                }

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    TTask const & task)
                -> void
                {
                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamAddCallback)(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue,
                        uniformCudaHipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));

                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call CUDA/HIP methods within the CUDA/HIP callback thread.
                    // The CUDA/HIP thread signals the std::thread when it is ready to execute the task.
                    // The CUDA/HIP thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (CUDA/HIP stream).
                    std::thread t(
                        [pCallbackSynchronizationData, task](){
                            // If the callback has not yet been called, we wait for it.
                            {
                                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                                if(pCallbackSynchronizationData->state != CallbackState::notified)
                                {
                                    pCallbackSynchronizationData->m_event.wait(
                                        lock,
                                        [pCallbackSynchronizationData](){
                                            return pCallbackSynchronizationData->state == CallbackState::notified;
                                        }
                                    );
                                }

                                task();

                                // Notify the waiting CUDA/HIP thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();
                        }
                    );

                    t.join();
                }
            };
            //#############################################################################
            //! The CUDA/HIP RT blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueUniformCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueUniformCudaHipRtBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for queues on non current device.
                    ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                        ret = ALPAKA_API_PREFIX(StreamQuery)(queue.m_spQueueImpl->m_UniformCudaHipQueue),
                        ALPAKA_API_PREFIX(ErrorNotReady));
                    return (ret == ALPAKA_API_PREFIX(Success));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueUniformCudaHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueUniformCudaHipRtBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for queues on non current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK( ALPAKA_API_PREFIX(StreamSynchronize)(
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            };
        }
    }
}

#endif
