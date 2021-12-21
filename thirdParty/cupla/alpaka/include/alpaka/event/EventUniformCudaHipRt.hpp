/* Copyright 2019 Benjamin Worpitz
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

#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

#    include <functional>
#    include <memory>
#    include <stdexcept>

namespace alpaka
{
    namespace uniform_cuda_hip
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA/HIP RT device event implementation.
            class EventUniformCudaHipImpl final
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST EventUniformCudaHipImpl(DevUniformCudaHipRt const& dev, bool bBusyWait)
                    : m_dev(dev)
                    , m_UniformCudaHipEvent()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.m_iDevice));

                    // Create the event on the current device with the specified flags. Valid flags include:
                    // - cuda/hip-EventDefault: Default event creation flag.
                    // - cuda/hip-EventBlockingSync : Specifies that event should use blocking synchronization.
                    //   A host thread that uses cuda/hip-EventSynchronize() to wait on an event created with this flag
                    //   will block until the event actually completes.
                    // - cuda/hip-EventDisableTiming : Specifies that the created event does not need to record timing
                    // data.
                    //   Events created with this flag specified and the cuda/hip-EventBlockingSync flag not specified
                    //   will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(EventCreateWithFlags)(
                        &m_UniformCudaHipEvent,
                        (bBusyWait ? ALPAKA_API_PREFIX(EventDefault) : ALPAKA_API_PREFIX(EventBlockingSync))
                            | ALPAKA_API_PREFIX(EventDisableTiming)));
                }
                //-----------------------------------------------------------------------------
                EventUniformCudaHipImpl(EventUniformCudaHipImpl const&) = delete;
                //-----------------------------------------------------------------------------
                EventUniformCudaHipImpl(EventUniformCudaHipImpl&&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(EventUniformCudaHipImpl const&) -> EventUniformCudaHipImpl& = delete;
                //-----------------------------------------------------------------------------
                auto operator=(EventUniformCudaHipImpl&&) -> EventUniformCudaHipImpl& = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST ~EventUniformCudaHipImpl()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device. \TODO: Is setting the current device before cuda/hip-EventDestroy
                    // required?

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.m_iDevice));
                    // In case event has been recorded but has not yet been completed when cuda/hip-EventDestroy() is
                    // called, the function will return immediately and the resources associated with event will be
                    // released automatically once the device has completed event.
                    // -> No need to synchronize here.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(EventDestroy)(m_UniformCudaHipEvent));
                }

            public:
                DevUniformCudaHipRt const m_dev; //!< The device this event is bound to.

                ALPAKA_API_PREFIX(Event_t) m_UniformCudaHipEvent;
            };
        } // namespace detail
    } // namespace uniform_cuda_hip

    //#############################################################################
    //! The CUDA/HIP RT device event.
    class EventUniformCudaHipRt final
        : public concepts::Implements<ConceptCurrentThreadWaitFor, EventUniformCudaHipRt>
        , public concepts::Implements<ConceptGetDev, EventUniformCudaHipRt>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST EventUniformCudaHipRt(DevUniformCudaHipRt const& dev, bool bBusyWait = true)
            : m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventUniformCudaHipImpl>(dev, bBusyWait))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }
        //-----------------------------------------------------------------------------
        EventUniformCudaHipRt(EventUniformCudaHipRt const&) = default;
        //-----------------------------------------------------------------------------
        EventUniformCudaHipRt(EventUniformCudaHipRt&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventUniformCudaHipRt const&) -> EventUniformCudaHipRt& = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventUniformCudaHipRt&&) -> EventUniformCudaHipRt& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(EventUniformCudaHipRt const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(EventUniformCudaHipRt const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~EventUniformCudaHipRt() = default;

    public:
        std::shared_ptr<uniform_cuda_hip::detail::EventUniformCudaHipImpl> m_spEventImpl;
    };
    namespace traits
    {
        //#############################################################################
        //! The CUDA/HIP RT device event device get trait specialization.
        template<>
        struct GetDev<EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(EventUniformCudaHipRt const& event) -> DevUniformCudaHipRt
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CUDA/HIP RT device event test trait specialization.
        template<>
        struct IsComplete<EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isComplete(EventUniformCudaHipRt const& event) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for events on non current device.
                ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = ALPAKA_API_PREFIX(EventQuery)(event.m_spEventImpl->m_UniformCudaHipEvent),
                    ALPAKA_API_PREFIX(ErrorNotReady));
                return (ret == ALPAKA_API_PREFIX(Success));
            }
        };

        //#############################################################################
        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueUniformCudaHipRtNonBlocking, EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtNonBlocking& queue, EventUniformCudaHipRt& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(EventRecord)(
                    event.m_spEventImpl->m_UniformCudaHipEvent,
                    queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };
        //#############################################################################
        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueUniformCudaHipRtBlocking, EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtBlocking& queue, EventUniformCudaHipRt& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(EventRecord)(
                    event.m_spEventImpl->m_UniformCudaHipEvent,
                    queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };

        //#############################################################################
        //! The CUDA/HIP RT device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        template<>
        struct CurrentThreadWaitFor<EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for events on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(EventSynchronize)(event.m_spEventImpl->m_UniformCudaHipEvent));
            }
        };
        //#############################################################################
        //! The CUDA/HIP RT queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueUniformCudaHipRtNonBlocking, EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtNonBlocking& queue,
                EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamWaitEvent)(
                    queue.m_spQueueImpl->m_UniformCudaHipQueue,
                    event.m_spEventImpl->m_UniformCudaHipEvent,
                    0));
            }
        };
        //#############################################################################
        //! The CUDA/HIP RT queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueUniformCudaHipRtBlocking, EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtBlocking& queue,
                EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamWaitEvent)(
                    queue.m_spQueueImpl->m_UniformCudaHipQueue,
                    event.m_spEventImpl->m_UniformCudaHipEvent,
                    0));
            }
        };
        //#############################################################################
        //! The CUDA/HIP RT device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning
        //! execution.
        template<>
        struct WaiterWaitFor<DevUniformCudaHipRt, EventUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(DevUniformCudaHipRt& dev, EventUniformCudaHipRt const& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.m_iDevice));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamWaitEvent)(nullptr, event.m_spEventImpl->m_UniformCudaHipEvent, 0));
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
