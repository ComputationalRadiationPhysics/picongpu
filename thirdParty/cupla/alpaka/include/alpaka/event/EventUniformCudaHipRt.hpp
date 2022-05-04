/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
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
    namespace uniform_cuda_hip::detail
    {
        //! The CUDA/HIP RT device event implementation.
        class EventUniformCudaHipImpl final
        {
        public:
            ALPAKA_FN_HOST EventUniformCudaHipImpl(DevUniformCudaHipRt const& dev, bool bBusyWait)
                : m_dev(dev)
                , m_UniformCudaHipEvent()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.getNativeHandle()));

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
            EventUniformCudaHipImpl(EventUniformCudaHipImpl const&) = delete;
            auto operator=(EventUniformCudaHipImpl const&) -> EventUniformCudaHipImpl& = delete;
            ALPAKA_FN_HOST ~EventUniformCudaHipImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // In case event has been recorded but has not yet been completed when cuda/hip-EventDestroy() is
                // called, the function will return immediately and the resources associated with event will be
                // released automatically once the device has completed event.
                // -> No need to synchronize here.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ALPAKA_API_PREFIX(EventDestroy)(m_UniformCudaHipEvent));
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipEvent;
            }

        public:
            DevUniformCudaHipRt const m_dev; //!< The device this event is bound to.

        private:
            ALPAKA_API_PREFIX(Event_t) m_UniformCudaHipEvent;
        };
    } // namespace uniform_cuda_hip::detail

    //! The CUDA/HIP RT device event.
    class EventUniformCudaHipRt final
        : public concepts::Implements<ConceptCurrentThreadWaitFor, EventUniformCudaHipRt>
        , public concepts::Implements<ConceptGetDev, EventUniformCudaHipRt>
    {
    public:
        ALPAKA_FN_HOST EventUniformCudaHipRt(DevUniformCudaHipRt const& dev, bool bBusyWait = true)
            : m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventUniformCudaHipImpl>(dev, bBusyWait))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }
        ALPAKA_FN_HOST auto operator==(EventUniformCudaHipRt const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }
        ALPAKA_FN_HOST auto operator!=(EventUniformCudaHipRt const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return m_spEventImpl->getNativeHandle();
        }

    public:
        std::shared_ptr<uniform_cuda_hip::detail::EventUniformCudaHipImpl> m_spEventImpl;
    };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using EventCudaRt = EventUniformCudaHipRt;
#    else
    using EventHipRt = EventUniformCudaHipRt;
#    endif

    namespace trait
    {
        //! The CUDA/HIP RT device event device type trait specialization.
        template<>
        struct DevType<EventUniformCudaHipRt>
        {
            using type = DevUniformCudaHipRt;
        };
        //! The CUDA/HIP RT device event device get trait specialization.
        template<>
        struct GetDev<EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto getDev(EventUniformCudaHipRt const& event) -> DevUniformCudaHipRt
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT device event test trait specialization.
        template<>
        struct IsComplete<EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto isComplete(EventUniformCudaHipRt const& event) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for events on non current device.
                ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = ALPAKA_API_PREFIX(EventQuery)(event.getNativeHandle()),
                    ALPAKA_API_PREFIX(ErrorNotReady));
                return (ret == ALPAKA_API_PREFIX(Success));
            }
        };

        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueUniformCudaHipRtNonBlocking, EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtNonBlocking& queue, EventUniformCudaHipRt& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(EventRecord)(event.getNativeHandle(), queue.getNativeHandle()));
            }
        };
        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueUniformCudaHipRtBlocking, EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtBlocking& queue, EventUniformCudaHipRt& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(EventRecord)(event.getNativeHandle(), queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        template<>
        struct CurrentThreadWaitFor<EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for events on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(EventSynchronize)(event.getNativeHandle()));
            }
        };
        //! The CUDA/HIP RT queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueUniformCudaHipRtNonBlocking, EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtNonBlocking& queue,
                EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamWaitEvent)(queue.getNativeHandle(), event.getNativeHandle(), 0));
            }
        };
        //! The CUDA/HIP RT queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueUniformCudaHipRtBlocking, EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtBlocking& queue,
                EventUniformCudaHipRt const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamWaitEvent)(queue.getNativeHandle(), event.getNativeHandle(), 0));
            }
        };
        //! The CUDA/HIP RT device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning
        //! execution.
        template<>
        struct WaiterWaitFor<DevUniformCudaHipRt, EventUniformCudaHipRt>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(DevUniformCudaHipRt& dev, EventUniformCudaHipRt const& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(dev.getNativeHandle()));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamWaitEvent)(nullptr, event.getNativeHandle(), 0));
            }
        };
        //! The CUDA/HIP RT event native handle trait specialization.
        template<>
        struct NativeHandle<EventUniformCudaHipRt>
        {
            [[nodiscard]] static auto getNativeHandle(EventUniformCudaHipRt const& event)
            {
                return event.getNativeHandle();
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
