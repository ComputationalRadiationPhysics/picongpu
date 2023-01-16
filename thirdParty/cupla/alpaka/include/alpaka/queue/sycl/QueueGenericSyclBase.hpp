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

#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <algorithm>
#    include <exception>
#    include <memory>
#    include <mutex>
#    include <shared_mutex>
#    include <type_traits>
#    include <utility>
#    include <vector>

namespace alpaka::experimental::detail
{
    template<typename T, typename = void>
    inline constexpr auto is_sycl_task = false;

    template<typename T>
    inline constexpr auto is_sycl_task<T, std::void_t<decltype(T::is_sycl_task)>> = true;

    template<typename T, typename = void>
    inline constexpr auto is_sycl_kernel = false;

    template<typename T>
    inline constexpr auto is_sycl_kernel<T, std::void_t<decltype(T::is_sycl_kernel)>> = true;

    class QueueGenericSyclImpl
    {
    public:
        QueueGenericSyclImpl(sycl::context context, sycl::device device)
            : m_queue{
                std::move(context), // This is important. In SYCL a device can belong to multiple contexts.
                std::move(device),
                {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}}
        {
        }

        // This class will only exist as a pointer. We don't care about copy and move semantics.
        QueueGenericSyclImpl(QueueGenericSyclImpl const& other) = delete;
        auto operator=(QueueGenericSyclImpl const& rhs) -> QueueGenericSyclImpl& = delete;

        QueueGenericSyclImpl(QueueGenericSyclImpl&& other) noexcept = delete;
        auto operator=(QueueGenericSyclImpl&& rhs) noexcept -> QueueGenericSyclImpl& = delete;

        ~QueueGenericSyclImpl()
        {
            try
            {
                m_queue.wait_and_throw();
            }
            catch(sycl::exception const& err)
            {
                std::cerr << "Caught SYCL exception while destructing a SYCL queue: " << err.what() << " ("
                          << err.code() << ')' << std::endl;
            }
            catch(std::exception const& err)
            {
                std::cerr << "The following runtime error(s) occured while destructing a SYCL queue:" << err.what()
                          << std::endl;
            }
        }

        // Don't call this without locking first!
        auto clean_dependencies() -> void
        {
            // Clean up completed events
            auto const start = std::begin(m_dependencies);
            auto const old_end = std::end(m_dependencies);
            auto const new_end = std::remove_if(
                start,
                old_end,
                [](sycl::event ev) {
                    return ev.get_info<sycl::info::event::command_execution_status>()
                           == sycl::info::event_command_status::complete;
                });

            m_dependencies.erase(new_end, old_end);
        }

        auto register_dependency(sycl::event event) -> void
        {
            std::lock_guard<std::shared_mutex> lock{m_mutex};

            clean_dependencies();
            m_dependencies.push_back(event);
        }

        auto empty() const -> bool
        {
            std::shared_lock<std::shared_mutex> lock{m_mutex};
            return m_last_event.get_info<sycl::info::event::command_execution_status>()
                   == sycl::info::event_command_status::complete;
        }

        auto wait() -> void
        {
            // SYCL queues are thread-safe.
            m_queue.wait_and_throw();
        }

        auto get_last_event() const -> sycl::event
        {
            std::shared_lock<std::shared_mutex> lock{m_mutex};
            return m_last_event;
        }

        template<bool TBlocking, typename TTask>
        auto enqueue(TTask const& task) -> void
        {
            {
                std::lock_guard<std::shared_mutex> lock{m_mutex};

                clean_dependencies();

                // Execute task
                m_last_event = m_queue.submit(
                    [this, &task](sycl::handler& cgh)
                    {
                        if(!m_dependencies.empty())
                            cgh.depends_on(m_dependencies);

                        if constexpr(is_sycl_kernel<TTask>) // Kernel
                            task(cgh, m_fence_dummy); // Will call cgh.parallel_for internally
                        else if constexpr(is_sycl_task<TTask>) // Copy / Fill
                            task(cgh); // Will call cgh.{copy, fill} internally
                        else // Host
                            cgh.host_task(task);
                    });

                m_dependencies.clear();
            }

            if constexpr(TBlocking)
                wait();
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return m_queue;
        }

        std::vector<sycl::event> m_dependencies;
        sycl::event m_last_event;
        sycl::buffer<int, 1> m_fence_dummy{sycl::range<1>{1}};
        std::shared_mutex mutable m_mutex;

    private:
        sycl::queue m_queue;
    };

    template<typename TDev, bool TBlocking>
    class QueueGenericSyclBase
    {
    public:
        QueueGenericSyclBase(TDev const& dev)
            : m_dev{dev}
            , m_impl{std::make_shared<detail::QueueGenericSyclImpl>(
                  dev.getNativeHandle().second,
                  dev.getNativeHandle().first)}
        {
            m_dev.m_impl->register_queue(m_impl);
        }

        friend auto operator==(QueueGenericSyclBase const& lhs, QueueGenericSyclBase const& rhs) -> bool
        {
            return (lhs.m_dev == rhs.m_dev) && (lhs.m_impl == rhs.m_impl);
        }

        friend auto operator!=(QueueGenericSyclBase const& lhs, QueueGenericSyclBase const& rhs) -> bool
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return m_impl->getNativeHandle();
        }

        TDev m_dev;
        std::shared_ptr<detail::QueueGenericSyclImpl> m_impl;
    };
} // namespace alpaka::experimental::detail

namespace alpaka::experimental
{
    template<typename TDev>
    class EventGenericSycl;
}

namespace alpaka::trait
{
    //! The SYCL blocking queue device type trait specialization.
    template<typename TDev, bool TBlocking>
    struct DevType<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        using type = TDev;
    };

    //! The SYCL blocking queue device get trait specialization.
    template<typename TDev, bool TBlocking>
    struct GetDev<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto getDev(experimental::detail::QueueGenericSyclBase<TDev, TBlocking> const& queue)
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            return queue.m_dev;
        }
    };

    //! The SYCL blocking queue event type trait specialization.
    template<typename TDev, bool TBlocking>
    struct EventType<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        using type = experimental::EventGenericSycl<TDev>;
    };

    //! The SYCL blocking queue enqueue trait specialization.
    template<typename TDev, bool TBlocking, typename TTask>
    struct Enqueue<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>, TTask>
    {
        static auto enqueue(experimental::detail::QueueGenericSyclBase<TDev, TBlocking>& queue, TTask const& task)
            -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            queue.m_impl->template enqueue<TBlocking>(task);
        }
    };

    //! The SYCL blocking queue test trait specialization.
    template<typename TDev, bool TBlocking>
    struct Empty<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto empty(experimental::detail::QueueGenericSyclBase<TDev, TBlocking> const& queue) -> bool
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            return queue.m_impl->empty();
        }
    };

    //! The SYCL blocking queue thread wait trait specialization.
    //!
    //! Blocks execution of the calling thread until the queue has finished processing all previously requested
    //! tasks (kernels, data copies, ...)
    template<typename TDev, bool TBlocking>
    struct CurrentThreadWaitFor<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto currentThreadWaitFor(experimental::detail::QueueGenericSyclBase<TDev, TBlocking> const& queue)
            -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            queue.m_impl->wait();
        }
    };

    //! The SYCL queue native handle trait specialization.
    template<typename TDev, bool TBlocking>
    struct NativeHandle<experimental::detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        [[nodiscard]] static auto getNativeHandle(
            experimental::detail::QueueGenericSyclBase<TDev, TBlocking> const& queue)
        {
            return queue.getNativeHandle();
        }
    };
} // namespace alpaka::trait

#endif
