/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/DevHipRt.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <alpaka/core/Hip.hpp>

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
        class EventHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                //! The HIP RT non-blocking queue implementation.
                class QueueHipRtNonBlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueHipRtNonBlockingImpl(
                        dev::DevHipRt const & dev) :
                            m_dev(dev),
                            m_HipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // - hipStreamDefault: Default queue creation flag.
                        // - hipStreamNonBlocking: Specifies that work running in the created queue may run concurrently with work in queue 0 (the NULL queue),
                        //   and that the created queue should perform no implicit synchronization with queue 0.
                        // Create the queue on the current device.
                        // NOTE: hipStreamNonBlocking is required to match the semantic implemented in the alpaka CPU queue.
                        // It would be too much work to implement implicit default queue synchronization on CPU.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamCreateWithFlags(
                                &m_HipQueue,
                                hipStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    QueueHipRtNonBlockingImpl(QueueHipRtNonBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueHipRtNonBlockingImpl(QueueHipRtNonBlockingImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtNonBlockingImpl const &) -> QueueHipRtNonBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtNonBlockingImpl &&) -> QueueHipRtNonBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueHipRtNonBlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the queue when hipStreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                m_HipQueue));
                    }

                public:
                    dev::DevHipRt const m_dev;   //!< The device this queue is bound to.
                    hipStream_t m_HipQueue;

#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    int m_callees = 0;
                    std::mutex m_mutex;
#endif
                };
            }
        }

        //#############################################################################
        //! The HIP RT non-blocking queue.
        class QueueHipRtNonBlocking final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueHipRtNonBlocking>
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueHipRtNonBlocking(
                dev::DevHipRt const & dev) :
                m_spQueueImpl(std::make_shared<hip::detail::QueueHipRtNonBlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueHipRtNonBlocking(QueueHipRtNonBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueHipRtNonBlocking(QueueHipRtNonBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtNonBlocking const &) -> QueueHipRtNonBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtNonBlocking &&) -> QueueHipRtNonBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueHipRtNonBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueHipRtNonBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~QueueHipRtNonBlocking() {
#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                // we are a non-blocking queue, so we have to wait here with its destruction until all spawned tasks have been processed
                alpaka::wait::wait(*this);
#endif
            }

        public:
            std::shared_ptr<hip::detail::QueueHipRtNonBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT non-blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueHipRtNonBlocking>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The HIP RT non-blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueHipRtNonBlocking const & queue)
                -> dev::DevHipRt
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
            //! The HIP RT non-blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueHipRtNonBlocking>
            {
                using type = event::EventHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueHipRtNonBlocking,
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
                static void HIPRT_CB hipRtCallback(hipStream_t /*queue*/, hipError_t /*status*/, void *arg)
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
                    queue::QueueHipRtNonBlocking & queue,
                    TTask const & task)
                -> void
                {
#if BOOST_COMP_HIP
                    // NOTE: hip callbacks are not blocking the stream.
                    // The workaround used for HIP(hcc) would avoid the usage in a workflow with
                    // many stream/event synchronizations (e.g. PIConGPU).
                    // @todo remove this assert when hipStreamAddCallback is fixed
                    static_assert(
                                meta::DependentFalseType<TTask>::value,
                                "Callbacks are not supported for HIP-clang");
#endif

#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    {
                        // thread-safe callee incrementing
                        std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);
                        queue.m_spQueueImpl->m_callees += 1;
                    }
#endif
                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();
                    // test example: https://github.com/ROCm-Developer-Tools/HIP/blob/roc-1.9.x/tests/src/runtimeApi/stream/hipStreamAddCallback.cpp
                    ALPAKA_HIP_RT_CHECK(hipStreamAddCallback(
                        queue.m_spQueueImpl->m_HipQueue,
                        hipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));

                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call HIP methods within the HIP callback thread.
                    // The HIP thread signals the std::thread when it is ready to execute the task.
                    // The HIP thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (HIP stream).
                    std::thread t(
                        [pCallbackSynchronizationData,
                         task
#if BOOST_COMP_HCC // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                         ,&queue // requires queue's destructor to wait for all tasks
#endif
                        ](){

#if BOOST_COMP_HCC // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                            // thread-safe task execution and callee decrementing
                            std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);
#endif

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

                                // Notify the waiting HIP thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();
#if BOOST_COMP_HCC // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                            queue.m_spQueueImpl->m_callees -= 1;
#endif
                        }
                    );

                    t.detach();
                }
            };
            //#############################################################################
            //! The HIP RT non-blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueHipRtNonBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    return (queue.m_spQueueImpl->m_callees==0);
#else

                    // Query is allowed even for queues on non current device.
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipStreamQuery(
                            queue.m_spQueueImpl->m_HipQueue),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
#endif
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT non-blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueHipRtNonBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if BOOST_COMP_HCC  // NOTE: workaround for unwanted nonblocking hip streams for HCC (NVCC streams are blocking)
                    while(queue.m_spQueueImpl->m_callees>0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10u));
                    }
#else
                    // Sync is allowed even for queues on non current device.
                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                            queue.m_spQueueImpl->m_HipQueue));
#endif
                }
            };
        }
    }
}

#endif
