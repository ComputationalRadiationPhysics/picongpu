/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/core/Unused.hpp>

#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>

#include <type_traits>
#include <thread>
#include <mutex>
#include <future>

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
                class QueueGenericThreadsNonBlockingImpl final : public IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                private:
                    //#############################################################################
                    using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
                        std::size_t,
                        std::thread,                // The concurrent execution type.
                        std::promise,               // The promise type.
                        void,                       // The type yielding the current concurrent execution.
                        std::mutex,                 // The mutex type to use. Only required if TisYielding is true.
                        std::condition_variable,    // The condition variable type to use. Only required if TisYielding is true.
                        false>;                     // If the threads should yield.

                public:
                    //-----------------------------------------------------------------------------
                    explicit QueueGenericThreadsNonBlockingImpl(
                        TDev const & dev) :
                            m_dev(dev),
                            m_workerThread(1u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev> const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev> &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericThreadsNonBlockingImpl<TDev> const &) -> QueueGenericThreadsNonBlockingImpl<TDev> & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueGenericThreadsNonBlockingImpl<TDev> &&) -> QueueGenericThreadsNonBlockingImpl<TDev> & = delete;

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

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        template<
            typename TDev>
        class QueueGenericThreadsNonBlocking final
            : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueGenericThreadsNonBlocking<TDev>>
            , public concepts::Implements<ConceptQueue, QueueGenericThreadsNonBlocking<TDev>>
            , public concepts::Implements<dev::ConceptGetDev, QueueGenericThreadsNonBlocking<TDev>>
        {
        public:
            //-----------------------------------------------------------------------------
            explicit QueueGenericThreadsNonBlocking(
                TDev const & dev) :
                    m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>>(dev))
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                dev.registerQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueGenericThreadsNonBlocking(QueueGenericThreadsNonBlocking<TDev> const &) = default;
            //-----------------------------------------------------------------------------
            QueueGenericThreadsNonBlocking(QueueGenericThreadsNonBlocking<TDev> &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericThreadsNonBlocking<TDev> const &) -> QueueGenericThreadsNonBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueGenericThreadsNonBlocking<TDev> &&) -> QueueGenericThreadsNonBlocking<TDev> & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueGenericThreadsNonBlocking<TDev> const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueGenericThreadsNonBlocking<TDev> const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueGenericThreadsNonBlocking() = default;

        public:
            std::shared_ptr<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU non-blocking device queue device type trait specialization.
            template<
                typename TDev>
            struct DevType<
                queue::QueueGenericThreadsNonBlocking<TDev>>
            {
                using type = TDev;
            };
            //#############################################################################
            //! The CPU non-blocking device queue device get trait specialization.
            template<
                typename TDev>
            struct GetDev<
                queue::QueueGenericThreadsNonBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueGenericThreadsNonBlocking<TDev> const & queue)
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
            //! The CPU non-blocking device queue event type trait specialization.
            template<
                typename TDev>
            struct EventType<
                queue::QueueGenericThreadsNonBlocking<TDev>>
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
            //! The CPU non-blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TDev,
                typename TTask>
            struct Enqueue<
                queue::QueueGenericThreadsNonBlocking<TDev>,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueGenericThreadsNonBlocking<TDev> & queue,
                    TTask const & task)
                -> void
                {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue.m_spQueueImpl->m_workerThread.enqueueTask(
                        task);
#else
                    alpaka::ignore_unused(queue);
                    alpaka::ignore_unused(task);
#endif
                }
            };
            //#############################################################################
            //! The CPU non-blocking device queue test trait specialization.
            template<
                typename TDev>
            struct Empty<
                queue::QueueGenericThreadsNonBlocking<TDev>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueGenericThreadsNonBlocking<TDev> const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_workerThread.isIdle();
                }
            };
        }
    }
}

#include <alpaka/event/EventGenericThreads.hpp>
