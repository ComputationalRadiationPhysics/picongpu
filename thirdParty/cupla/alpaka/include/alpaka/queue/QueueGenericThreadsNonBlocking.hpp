/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/wait/Traits.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace alpaka
{
    template<typename TDev>
    class EventGenericThreads;

    namespace generic
    {
        namespace detail
        {
#if BOOST_COMP_CLANG
// avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every
// translation unit [-Werror,-Wweak-vtables]" https://stackoverflow.com/a/29288300
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
            //! The CPU device queue implementation.
            template<typename TDev>
            class QueueGenericThreadsNonBlockingImpl final : public IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
            {
            private:
                using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
                    std::size_t,
                    std::thread, // The concurrent execution type.
                    std::promise, // The promise type.
                    void, // The type yielding the current concurrent execution.
                    std::mutex, // The mutex type to use. Only required if TisYielding is true.
                    std::condition_variable, // The condition variable type to use. Only required if TisYielding is
                                             // true.
                    false>; // If the threads should yield.

            public:
                explicit QueueGenericThreadsNonBlockingImpl(TDev dev)
                    : m_dev(std::move(dev))
                    , m_workerThread(std::make_shared<ThreadPool>(1u))
                {
                }
                QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev> const&) = delete;
                QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev>&&) = delete;
                auto operator=(QueueGenericThreadsNonBlockingImpl<TDev> const&)
                    -> QueueGenericThreadsNonBlockingImpl<TDev>& = delete;
                auto operator=(QueueGenericThreadsNonBlockingImpl&&)
                    -> QueueGenericThreadsNonBlockingImpl<TDev>& = delete;
                ~QueueGenericThreadsNonBlockingImpl() override
                {
                    m_dev.registerCleanup(
                        [pool = std::weak_ptr<ThreadPool>(m_workerThread)]() noexcept
                        {
                            if(auto s = pool.lock())
                                static_cast<void>(s->takeDetachHandle()); // let returned shared_ptr destroy itself
                        });
                    auto* wt = m_workerThread.get();
                    wt->detach(std::move(m_workerThread));
                }

                void enqueue(EventGenericThreads<TDev>& ev) final
                {
                    alpaka::enqueue(*this, ev);
                }

                void wait(EventGenericThreads<TDev> const& ev) final
                {
                    alpaka::wait(*this, ev);
                }

            public:
                TDev const m_dev; //!< The device this queue is bound to.

                std::shared_ptr<ThreadPool> m_workerThread;
            };
        } // namespace detail
    } // namespace generic

    //! The CPU device queue.
    template<typename TDev>
    class QueueGenericThreadsNonBlocking final
        : public concepts::Implements<ConceptCurrentThreadWaitFor, QueueGenericThreadsNonBlocking<TDev>>
        , public concepts::Implements<ConceptQueue, QueueGenericThreadsNonBlocking<TDev>>
        , public concepts::Implements<ConceptGetDev, QueueGenericThreadsNonBlocking<TDev>>
    {
    public:
        explicit QueueGenericThreadsNonBlocking(TDev const& dev)
            : m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>>(dev))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            dev.registerQueue(m_spQueueImpl);
        }
        auto operator==(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
        {
            return (m_spQueueImpl == rhs.m_spQueueImpl);
        }
        auto operator!=(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

    public:
        std::shared_ptr<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>> m_spQueueImpl;
    };

    namespace trait
    {
        //! The CPU non-blocking device queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueGenericThreadsNonBlocking<TDev>>
        {
            using type = TDev;
        };
        //! The CPU non-blocking device queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueGenericThreadsNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(QueueGenericThreadsNonBlocking<TDev> const& queue) -> TDev
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CPU non-blocking device queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueGenericThreadsNonBlocking<TDev>>
        {
            using type = EventGenericThreads<TDev>;
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(
                [[maybe_unused]] QueueGenericThreadsNonBlocking<TDev>& queue,
                [[maybe_unused]] TTask const& task) -> void
            {
                // Workaround: Clang can not support this when natively compiling device code. See
                // ConcurrentExecPool.hpp.
                if constexpr(!((BOOST_COMP_CLANG_CUDA != BOOST_VERSION_NUMBER_NOT_AVAILABLE)
                               && (BOOST_ARCH_PTX != BOOST_VERSION_NUMBER_NOT_AVAILABLE)))
                    queue.m_spQueueImpl->m_workerThread->enqueueTask(task);
            }
        };
        //! The CPU non-blocking device queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueGenericThreadsNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto empty(QueueGenericThreadsNonBlocking<TDev> const& queue) -> bool
            {
                return queue.m_spQueueImpl->m_workerThread->isIdle();
            }
        };
    } // namespace trait
} // namespace alpaka

#include <alpaka/event/EventGenericThreads.hpp>
