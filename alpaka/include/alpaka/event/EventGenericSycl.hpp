/* Copyright 2024 Jan Stephan, Antonio Di Pilato, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/queue/QueueGenericSyclBlocking.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"
#include "alpaka/wait/Traits.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL device event.
    template<typename TTag>
    class EventGenericSycl final
    {
    public:
        explicit EventGenericSycl(DevGenericSycl<TTag> const& dev) : m_dev{dev}
        {
        }

        friend auto operator==(EventGenericSycl const& lhs, EventGenericSycl const& rhs) -> bool
        {
            return (lhs.m_event == rhs.m_event);
        }

        friend auto operator!=(EventGenericSycl const& lhs, EventGenericSycl const& rhs) -> bool
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return m_event;
        }

        void setEvent(sycl::event const& event)
        {
            m_event = event;
        }

        DevGenericSycl<TTag> m_dev;

    private:
        sycl::event m_event{};
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device event device get trait specialization.
    template<typename TTag>
    struct GetDev<EventGenericSycl<TTag>>
    {
        static auto getDev(EventGenericSycl<TTag> const& event) -> DevGenericSycl<TTag>
        {
            return event.m_dev;
        }
    };

    //! The SYCL device event test trait specialization.
    template<typename TTag>
    struct IsComplete<EventGenericSycl<TTag>>
    {
        static auto isComplete(EventGenericSycl<TTag> const& event)
        {
            auto const status
                = event.getNativeHandle().template get_info<sycl::info::event::command_execution_status>();
            return (status == sycl::info::event_command_status::complete);
        }
    };

    //! The SYCL queue enqueue trait specialization.
    template<typename TTag>
    struct Enqueue<QueueGenericSyclNonBlocking<TTag>, EventGenericSycl<TTag>>
    {
        static auto enqueue(QueueGenericSyclNonBlocking<TTag>& queue, EventGenericSycl<TTag>& event)
        {
            event.setEvent(queue.m_spQueueImpl->get_last_event());
        }
    };

    //! The SYCL queue enqueue trait specialization.
    template<typename TTag>
    struct Enqueue<QueueGenericSyclBlocking<TTag>, EventGenericSycl<TTag>>
    {
        static auto enqueue(QueueGenericSyclBlocking<TTag>& queue, EventGenericSycl<TTag>& event)
        {
            event.setEvent(queue.m_spQueueImpl->get_last_event());
        }
    };

    //! The SYCL device event thread wait trait specialization.
    //!
    //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
    //! completed. If the event is not enqueued to a queue the method returns immediately.
    template<typename TTag>
    struct CurrentThreadWaitFor<EventGenericSycl<TTag>>
    {
        static auto currentThreadWaitFor(EventGenericSycl<TTag> const& event)
        {
            event.getNativeHandle().wait_and_throw();
        }
    };

    //! The SYCL queue event wait trait specialization.
    template<typename TTag>
    struct WaiterWaitFor<QueueGenericSyclNonBlocking<TTag>, EventGenericSycl<TTag>>
    {
        static auto waiterWaitFor(QueueGenericSyclNonBlocking<TTag>& queue, EventGenericSycl<TTag> const& event)
        {
            queue.m_spQueueImpl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL queue event wait trait specialization.
    template<typename TTag>
    struct WaiterWaitFor<QueueGenericSyclBlocking<TTag>, EventGenericSycl<TTag>>
    {
        static auto waiterWaitFor(QueueGenericSyclBlocking<TTag>& queue, EventGenericSycl<TTag> const& event)
        {
            queue.m_spQueueImpl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL device event wait trait specialization.
    //!
    //! Any future work submitted in any queue of this device will wait for event to complete before beginning
    //! execution.
    template<typename TTag>
    struct WaiterWaitFor<DevGenericSycl<TTag>, EventGenericSycl<TTag>>
    {
        static auto waiterWaitFor(DevGenericSycl<TTag>& dev, EventGenericSycl<TTag> const& event)
        {
            dev.m_impl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL device event native handle trait specialization.
    template<typename TTag>
    struct NativeHandle<EventGenericSycl<TTag>>
    {
        [[nodiscard]] static auto getNativeHandle(EventGenericSycl<TTag> const& event)
        {
            return event.getNativeHandle();
        }
    };
} // namespace alpaka::trait

#endif
