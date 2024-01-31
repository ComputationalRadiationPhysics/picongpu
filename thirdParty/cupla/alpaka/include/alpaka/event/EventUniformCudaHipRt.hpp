/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "alpaka/wait/Traits.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    namespace uniform_cuda_hip::detail
    {
        //! The CUDA/HIP RT device event implementation.
        template<typename TApi>
        class EventUniformCudaHipImpl final
        {
        public:
            ALPAKA_FN_HOST EventUniformCudaHipImpl(DevUniformCudaHipRt<TApi> const& dev, bool bBusyWait)
                : m_dev(dev)
                , m_UniformCudaHipEvent()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_dev.getNativeHandle()));

                // Create the event on the current device with the specified flags. Valid flags include:
                // - cuda/hip-EventDefault: Default event creation flag.
                // - cuda/hip-EventBlockingSync : Specifies that event should use blocking synchronization.
                //   A host thread that uses cuda/hip-EventSynchronize() to wait on an event created with this flag
                //   will block until the event actually completes.
                // - cuda/hip-EventDisableTiming : Specifies that the created event does not need to record timing
                // data.
                //   Events created with this flag specified and the cuda/hip-EventBlockingSync flag not specified
                //   will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventCreateWithFlags(
                    &m_UniformCudaHipEvent,
                    (bBusyWait ? TApi::eventDefault : TApi::eventBlockingSync) | TApi::eventDisableTiming));
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
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::eventDestroy(m_UniformCudaHipEvent));
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipEvent;
            }

        public:
            DevUniformCudaHipRt<TApi> const m_dev; //!< The device this event is bound to.

        private:
            typename TApi::Event_t m_UniformCudaHipEvent;
        };
    } // namespace uniform_cuda_hip::detail

    //! The CUDA/HIP RT device event.
    template<typename TApi>
    class EventUniformCudaHipRt final
        : public concepts::Implements<ConceptCurrentThreadWaitFor, EventUniformCudaHipRt<TApi>>
        , public concepts::Implements<ConceptGetDev, EventUniformCudaHipRt<TApi>>
    {
    public:
        ALPAKA_FN_HOST EventUniformCudaHipRt(DevUniformCudaHipRt<TApi> const& dev, bool bBusyWait = true)
            : m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventUniformCudaHipImpl<TApi>>(dev, bBusyWait))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }

        ALPAKA_FN_HOST auto operator==(EventUniformCudaHipRt<TApi> const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }

        ALPAKA_FN_HOST auto operator!=(EventUniformCudaHipRt<TApi> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return m_spEventImpl->getNativeHandle();
        }

    public:
        std::shared_ptr<uniform_cuda_hip::detail::EventUniformCudaHipImpl<TApi>> m_spEventImpl;
    };

    namespace trait
    {
        //! The CUDA/HIP RT device event device type trait specialization.
        template<typename TApi>
        struct DevType<EventUniformCudaHipRt<TApi>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT device event device get trait specialization.
        template<typename TApi>
        struct GetDev<EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getDev(EventUniformCudaHipRt<TApi> const& event) -> DevUniformCudaHipRt<TApi>
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT device event test trait specialization.
        template<typename TApi>
        struct IsComplete<EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto isComplete(EventUniformCudaHipRt<TApi> const& event) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for events on non current device.
                typename TApi::Error_t ret = TApi::success;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = TApi::eventQuery(event.getNativeHandle()),
                    TApi::errorNotReady);
                return (ret == TApi::success);
            }
        };

        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<typename TApi>
        struct Enqueue<QueueUniformCudaHipRtNonBlocking<TApi>, EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                EventUniformCudaHipRt<TApi>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventRecord(event.getNativeHandle(), queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<typename TApi>
        struct Enqueue<QueueUniformCudaHipRtBlocking<TApi>, EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                EventUniformCudaHipRt<TApi>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventRecord(event.getNativeHandle(), queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        template<typename TApi>
        struct CurrentThreadWaitFor<EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventUniformCudaHipRt<TApi> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for events on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventSynchronize(event.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT queue event wait trait specialization.
        template<typename TApi>
        struct WaiterWaitFor<QueueUniformCudaHipRtNonBlocking<TApi>, EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                EventUniformCudaHipRt<TApi> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamWaitEvent(queue.getNativeHandle(), event.getNativeHandle(), 0));
            }
        };

        //! The CUDA/HIP RT queue event wait trait specialization.
        template<typename TApi>
        struct WaiterWaitFor<QueueUniformCudaHipRtBlocking<TApi>, EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                EventUniformCudaHipRt<TApi> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamWaitEvent(queue.getNativeHandle(), event.getNativeHandle(), 0));
            }
        };

        //! The CUDA/HIP RT device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning
        //! execution.
        template<typename TApi>
        struct WaiterWaitFor<DevUniformCudaHipRt<TApi>, EventUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                DevUniformCudaHipRt<TApi>& dev,
                EventUniformCudaHipRt<TApi> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

                // Get all the queues on the device at the time of invocation.
                // All queues added afterwards are ignored.
                auto vQueues = dev.getAllQueues();
                for(auto&& spQueue : vQueues)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        TApi::streamWaitEvent(spQueue->getNativeHandle(), event.getNativeHandle(), 0));
                }
            }
        };

        //! The CUDA/HIP RT event native handle trait specialization.
        template<typename TApi>
        struct NativeHandle<EventUniformCudaHipRt<TApi>>
        {
            [[nodiscard]] static auto getNativeHandle(EventUniformCudaHipRt<TApi> const& event)
            {
                return event.getNativeHandle();
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
