/* Copyright 2023 Jan Stephan, Antonio Di Pilato, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::detail
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
                if constexpr(is_sycl_task<TTask> && !is_sycl_kernel<TTask>) // Copy / Fill
                {
                    m_last_event = task(m_queue, m_dependencies); // Will call queue.{copy, fill} internally
                }
                else
                {
                    m_last_event = m_queue.submit(
                        [this, &task](sycl::handler& cgh)
                        {
                            if(!m_dependencies.empty())
                                cgh.depends_on(m_dependencies);

                            if constexpr(is_sycl_kernel<TTask>) // Kernel
                                task(cgh); // Will call cgh.parallel_for internally
                            else // Host
                                cgh.host_task(task);
                        });
                }

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
            , m_spQueueImpl{std::make_shared<detail::QueueGenericSyclImpl>(
                  dev.getNativeHandle().second,
                  dev.getNativeHandle().first)}
        {
            m_dev.m_impl->register_queue(m_spQueueImpl);
        }

        friend auto operator==(QueueGenericSyclBase const& lhs, QueueGenericSyclBase const& rhs) -> bool
        {
            return (lhs.m_dev == rhs.m_dev) && (lhs.m_spQueueImpl == rhs.m_spQueueImpl);
        }

        friend auto operator!=(QueueGenericSyclBase const& lhs, QueueGenericSyclBase const& rhs) -> bool
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return m_spQueueImpl->getNativeHandle();
        }

        TDev m_dev;
        std::shared_ptr<detail::QueueGenericSyclImpl> m_spQueueImpl;
    };
} // namespace alpaka::detail

namespace alpaka
{
    template<typename TDev>
    class EventGenericSycl;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL blocking queue device type trait specialization.
    template<typename TDev, bool TBlocking>
    struct DevType<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        using type = TDev;
    };

    //! The SYCL blocking queue device get trait specialization.
    template<typename TDev, bool TBlocking>
    struct GetDev<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto getDev(detail::QueueGenericSyclBase<TDev, TBlocking> const& queue)
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            return queue.m_dev;
        }
    };

    //! The SYCL blocking queue event type trait specialization.
    template<typename TDev, bool TBlocking>
    struct EventType<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        using type = EventGenericSycl<TDev>;
    };

    //! The SYCL blocking queue enqueue trait specialization.
    template<typename TDev, bool TBlocking, typename TTask>
    struct Enqueue<detail::QueueGenericSyclBase<TDev, TBlocking>, TTask>
    {
        static auto enqueue(detail::QueueGenericSyclBase<TDev, TBlocking>& queue, TTask const& task) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            queue.m_spQueueImpl->template enqueue<TBlocking>(task);
        }
    };

    //! The SYCL blocking queue test trait specialization.
    template<typename TDev, bool TBlocking>
    struct Empty<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto empty(detail::QueueGenericSyclBase<TDev, TBlocking> const& queue) -> bool
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            return queue.m_spQueueImpl->empty();
        }
    };

    //! The SYCL blocking queue thread wait trait specialization.
    //!
    //! Blocks execution of the calling thread until the queue has finished processing all previously requested
    //! tasks (kernels, data copies, ...)
    template<typename TDev, bool TBlocking>
    struct CurrentThreadWaitFor<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        static auto currentThreadWaitFor(detail::QueueGenericSyclBase<TDev, TBlocking> const& queue) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            queue.m_spQueueImpl->wait();
        }
    };

    //! The SYCL queue native handle trait specialization.
    template<typename TDev, bool TBlocking>
    struct NativeHandle<detail::QueueGenericSyclBase<TDev, TBlocking>>
    {
        [[nodiscard]] static auto getNativeHandle(detail::QueueGenericSyclBase<TDev, TBlocking> const& queue)
        {
            return queue.getNativeHandle();
        }
    };
} // namespace alpaka::trait

#endif
