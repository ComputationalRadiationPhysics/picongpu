/* Copyright 2022 Jan Stephan, Antonio Di Pilato
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#    include <alpaka/wait/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <functional>
#    include <memory>
#    include <stdexcept>

namespace alpaka::experimental
{
    //! The SYCL device event.
    template<typename TDev>
    class EventGenericSycl final
    {
    public:
        explicit EventGenericSycl(TDev const& dev) : m_dev{dev}
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

        TDev m_dev;

    private:
        sycl::event m_event{};
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The SYCL device event device get trait specialization.
    template<typename TDev>
    struct GetDev<experimental::EventGenericSycl<TDev>>
    {
        static auto getDev(experimental::EventGenericSycl<TDev> const& event) -> TDev
        {
            return event.m_dev;
        }
    };

    //! The SYCL device event test trait specialization.
    template<typename TDev>
    struct IsComplete<experimental::EventGenericSycl<TDev>>
    {
        static auto isComplete(experimental::EventGenericSycl<TDev> const& event)
        {
            auto const status
                = event.getNativeHandle().template get_info<sycl::info::event::command_execution_status>();
            return (status == sycl::info::event_command_status::complete);
        }
    };

    //! The SYCL queue enqueue trait specialization.
    template<typename TDev>
    struct Enqueue<experimental::QueueGenericSyclNonBlocking<TDev>, experimental::EventGenericSycl<TDev>>
    {
        static auto enqueue(
            experimental::QueueGenericSyclNonBlocking<TDev>& queue,
            experimental::EventGenericSycl<TDev>& event)
        {
            event.setEvent(queue.m_impl->get_last_event());
        }
    };

    //! The SYCL queue enqueue trait specialization.
    template<typename TDev>
    struct Enqueue<experimental::QueueGenericSyclBlocking<TDev>, experimental::EventGenericSycl<TDev>>
    {
        static auto enqueue(
            experimental::QueueGenericSyclBlocking<TDev>& queue,
            experimental::EventGenericSycl<TDev>& event)
        {
            event.setEvent(queue.m_impl->get_last_event());
        }
    };

    //! The SYCL device event thread wait trait specialization.
    //!
    //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
    //! completed. If the event is not enqueued to a queue the method returns immediately.
    template<typename TDev>
    struct CurrentThreadWaitFor<experimental::EventGenericSycl<TDev>>
    {
        static auto currentThreadWaitFor(experimental::EventGenericSycl<TDev> const& event)
        {
            event.getNativeHandle().wait_and_throw();
        }
    };

    //! The SYCL queue event wait trait specialization.
    template<typename TDev>
    struct WaiterWaitFor<experimental::QueueGenericSyclNonBlocking<TDev>, experimental::EventGenericSycl<TDev>>
    {
        static auto waiterWaitFor(
            experimental::QueueGenericSyclNonBlocking<TDev>& queue,
            experimental::EventGenericSycl<TDev> const& event)
        {
            queue.m_impl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL queue event wait trait specialization.
    template<typename TDev>
    struct WaiterWaitFor<experimental::QueueGenericSyclBlocking<TDev>, experimental::EventGenericSycl<TDev>>
    {
        static auto waiterWaitFor(
            experimental::QueueGenericSyclBlocking<TDev>& queue,
            experimental::EventGenericSycl<TDev> const& event)
        {
            queue.m_impl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL device event wait trait specialization.
    //!
    //! Any future work submitted in any queue of this device will wait for event to complete before beginning
    //! execution.
    template<typename TDev>
    struct WaiterWaitFor<TDev, experimental::EventGenericSycl<TDev>>
    {
        static auto waiterWaitFor(TDev& dev, experimental::EventGenericSycl<TDev> const& event)
        {
            dev.m_impl->register_dependency(event.getNativeHandle());
        }
    };

    //! The SYCL device event native handle trait specialization.
    template<TDev>
    struct NativeHandle<experimental::EventGenericSycl<TDev>>
    {
        [[nodiscard]] static auto getNativeHandle(experimental::EventGenericSycl<TDev> const& event)
        {
            return event.getNativeHandle();
        }
    };
} // namespace alpaka::trait

#endif
