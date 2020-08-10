/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>

#include <atomic>
#include <mutex>
#include <memory>

namespace alpaka
{
    namespace event
    {
        template<typename TDev>
        class EventGenericThreads;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace generic
        {
            namespace detail
            {
#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif
                //#############################################################################
                //! The CPU device queue implementation.
                template<
                    typename TDev>
                class QueueGenericThreadsBlockingImpl final : public IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                public:
                    //-----------------------------------------------------------------------------
                    explicit QueueGenericThreadsBlockingImpl(
                        TDev const & dev) noexcept :
                            m_dev(dev),
                            m_bCurrentlyExecutingTask(false)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueGenericThreadsBlockingImpl(QueueGenericThreadsBlockingImpl<TDev> const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueGenericThreadsBlockingImpl(QueueGenericThreadsBlockingImpl<TDev> &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericThreadsBlockingImpl<TDev> const &) -> QueueGenericThreadsBlockingImpl<TDev> & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericThreadsBlockingImpl<TDev> &&) -> QueueGenericThreadsBlockingImpl<TDev> & = delete;

                    //-----------------------------------------------------------------------------
                    void enqueue(event::EventGenericThreads<TDev> & ev) final
                    {
                        queue::enqueue(*this, ev);
                    }

                    //-----------------------------------------------------------------------------
                    void wait(event::EventGenericThreads<TDev> const & ev) final
                    {
                        wait::wait(*this, ev);
                    }

                public:
                    TDev const m_dev;            //!< The device this queue is bound to.
                    std::mutex mutable m_mutex;
                    std::atomic<bool> m_bCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        template<
            typename TDev>
        class QueueGenericThreadsBlocking final
            : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueGenericThreadsBlocking<TDev>>
            , public concepts::Implements<ConceptQueue, QueueGenericThreadsBlocking<TDev>>
            , public concepts::Implements<dev::ConceptGetDev, QueueGenericThreadsBlocking<TDev>>
        {
        public:
            //-----------------------------------------------------------------------------
            explicit QueueGenericThreadsBlocking(
                TDev const & dev) :
                    m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsBlockingImpl<TDev>>(dev))
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                dev.registerQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueGenericThreadsBlocking(QueueGenericThreadsBlocking<TDev> const &) = default;
            //-----------------------------------------------------------------------------
            QueueGenericThreadsBlocking(QueueGenericThreadsBlocking<TDev> &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericThreadsBlocking<TDev> const &) -> QueueGenericThreadsBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericThreadsBlocking<TDev> &&) -> QueueGenericThreadsBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueGenericThreadsBlocking<TDev> const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueGenericThreadsBlocking<TDev> const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueGenericThreadsBlocking() = default;

        public:
            std::shared_ptr<generic::detail::QueueGenericThreadsBlockingImpl<TDev>> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue device type trait specialization.
            template<
                typename TDev>
            struct DevType<
                queue::QueueGenericThreadsBlocking<TDev>>
            {
                using type = TDev;
            };
            //#############################################################################
            //! The CPU blocking device queue device get trait specialization.
            template<
                typename TDev>
            struct GetDev<
                queue::QueueGenericThreadsBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueGenericThreadsBlocking<TDev> const & queue)
                -> TDev
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
            //! The CPU blocking device queue event type trait specialization.
            template<
                typename TDev>
            struct EventType<
                queue::QueueGenericThreadsBlocking<TDev>>
            {
                using type = event::EventGenericThreads<TDev>;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TDev,
                typename TTask>
            struct Enqueue<
                queue::QueueGenericThreadsBlocking<TDev>,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueGenericThreadsBlocking<TDev> & queue,
                    TTask const & task)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                    task();

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
                }
            };
            //#############################################################################
            //! The CPU blocking device queue test trait specialization.
            template<
                typename TDev>
            struct Empty<
                queue::QueueGenericThreadsBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueGenericThreadsBlocking<TDev> const & queue)
                -> bool
                {
                    return !queue.m_spQueueImpl->m_bCurrentlyExecutingTask;
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<
                typename TDev>
            struct CurrentThreadWaitFor<
                queue::QueueGenericThreadsBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueGenericThreadsBlocking<TDev> const & queue)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                }
            };
        }
    }
}

#include <alpaka/event/EventGenericThreads.hpp>
