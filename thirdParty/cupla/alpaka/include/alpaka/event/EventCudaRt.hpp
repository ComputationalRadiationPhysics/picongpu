/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/QueueCudaRtNonBlocking.hpp>
#include <alpaka/queue/QueueCudaRtBlocking.hpp>
#include <alpaka/core/Cuda.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    namespace event
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT device event implementation.
                class EventCudaImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaImpl(
                        dev::DevCudaRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_CudaEvent()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // Create the event on the current device with the specified flags. Valid flags include:
                        // - cudaEventDefault: Default event creation flag.
                        // - cudaEventBlockingSync : Specifies that event should use blocking synchronization.
                        //   A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                        // - cudaEventDisableTiming : Specifies that the created event does not need to record timing data.
                        //   Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                        ALPAKA_CUDA_RT_CHECK(
                            cudaEventCreateWithFlags(
                                &m_CudaEvent,
                                (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
                    }
                    //-----------------------------------------------------------------------------
                    EventCudaImpl(EventCudaImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventCudaImpl(EventCudaImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaImpl const &) -> EventCudaImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaImpl &&) -> EventCudaImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCudaImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaEventDestroy required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_dev.m_iDevice));
                        // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately
                        // and the resources associated with event will be released automatically once the device has completed event.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaEventDestroy(
                            m_CudaEvent));
                    }

                public:
                    dev::DevCudaRt const m_dev;   //!< The device this event is bound to.
                    cudaEvent_t m_CudaEvent;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT device event.
        class EventCudaRt final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaRt(
                dev::DevCudaRt const & dev,
                bool bBusyWait = true) :
                    m_spEventImpl(std::make_shared<cuda::detail::EventCudaImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventCudaRt(EventCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            EventCudaRt(EventCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaRt const &) -> EventCudaRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaRt &&) -> EventCudaRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCudaRt const & rhs) const
            -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCudaRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventCudaRt() = default;

        public:
            std::shared_ptr<cuda::detail::EventCudaImpl> m_spEventImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCudaRt const & event)
                -> dev::DevCudaRt
                {
                    return event.m_spEventImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event test trait specialization.
            template<>
            struct Test<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventCudaRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaEventQuery(
                            event.m_spEventImpl->m_CudaEvent),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaRtNonBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtNonBlocking & queue,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_CudaEvent,
                        queue.m_spQueueImpl->m_CudaQueue));
                }
            };
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking & queue,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_CudaEvent,
                        queue.m_spQueueImpl->m_CudaQueue));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        event.m_spEventImpl->m_CudaEvent));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaRtNonBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaRtNonBlocking & queue,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaQueue,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaRtBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaRtBlocking & queue,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaQueue,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevCudaRt,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaRt & dev,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        nullptr,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
        }
    }
}

#endif
