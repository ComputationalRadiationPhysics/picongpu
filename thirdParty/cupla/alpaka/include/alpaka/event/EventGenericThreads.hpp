/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/wait/Traits.hpp>

#include <condition_variable>
#include <future>
#include <mutex>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

namespace alpaka
{
    namespace generic
    {
        namespace detail
        {
            //#############################################################################
            //! The CPU device event implementation.
            template<typename TDev>
            class EventGenericThreadsImpl final
                : public concepts::Implements<ConceptCurrentThreadWaitFor, EventGenericThreadsImpl<TDev>>
            {
            public:
                //-----------------------------------------------------------------------------
                EventGenericThreadsImpl(TDev const& dev) noexcept
                    : m_dev(dev)
                    , m_mutex()
                    , m_enqueueCount(0u)
                    , m_LastReadyEnqueueCount(0u)
                {
                }
                //-----------------------------------------------------------------------------
                EventGenericThreadsImpl(EventGenericThreadsImpl<TDev> const&) = delete;
                //-----------------------------------------------------------------------------
                EventGenericThreadsImpl(EventGenericThreadsImpl<TDev>&&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(EventGenericThreadsImpl<TDev> const&) -> EventGenericThreadsImpl<TDev>& = delete;
                //-----------------------------------------------------------------------------
                auto operator=(EventGenericThreadsImpl<TDev>&&) -> EventGenericThreadsImpl<TDev>& = delete;
                //-----------------------------------------------------------------------------
                ~EventGenericThreadsImpl() noexcept = default;

                //-----------------------------------------------------------------------------
                auto isReady() noexcept -> bool
                {
                    return (m_LastReadyEnqueueCount == m_enqueueCount);
                }

                //-----------------------------------------------------------------------------
                auto wait(std::size_t const& enqueueCount, std::unique_lock<std::mutex>& lk) const noexcept -> void
                {
                    ALPAKA_ASSERT(enqueueCount <= m_enqueueCount);

                    while(enqueueCount > m_LastReadyEnqueueCount)
                    {
                        auto future = m_future;
                        lk.unlock();
                        future.get();
                        lk.lock();
                    }
                }

            public:
                TDev const m_dev; //!< The device this event is bound to.

                std::mutex mutable m_mutex; //!< The mutex used to synchronize access to the event.
                std::shared_future<void> m_future; //!< The future signaling the event completion.
                std::size_t m_enqueueCount; //!< The number of times this event has been enqueued.
                std::size_t m_LastReadyEnqueueCount; //!< The time this event has been ready the last time.
                                                     //!< Ready means that the event was not waiting within a queue
                                                     //!< (not enqueued or already completed). If m_enqueueCount ==
                                                     //!< m_LastReadyEnqueueCount, the event is currently not enqueued
            };
        } // namespace detail
    } // namespace generic

    //#############################################################################
    //! The CPU device event.
    template<typename TDev>
    class EventGenericThreads final
        : public concepts::Implements<ConceptCurrentThreadWaitFor, EventGenericThreads<TDev>>
        , public concepts::Implements<ConceptGetDev, EventGenericThreads<TDev>>
    {
    public:
        //-----------------------------------------------------------------------------
        //! \param bBusyWaiting Unused. EventGenericThreads never does busy waiting.
        EventGenericThreads(TDev const& dev, bool bBusyWaiting = true)
            : m_spEventImpl(std::make_shared<generic::detail::EventGenericThreadsImpl<TDev>>(dev))
        {
            alpaka::ignore_unused(bBusyWaiting);
        }
        //-----------------------------------------------------------------------------
        EventGenericThreads(EventGenericThreads<TDev> const&) = default;
        //-----------------------------------------------------------------------------
        EventGenericThreads(EventGenericThreads<TDev>&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericThreads<TDev> const&) -> EventGenericThreads<TDev>& = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericThreads<TDev>&&) -> EventGenericThreads<TDev>& = default;
        //-----------------------------------------------------------------------------
        auto operator==(EventGenericThreads<TDev> const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }
        //-----------------------------------------------------------------------------
        auto operator!=(EventGenericThreads<TDev> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~EventGenericThreads() = default;

    public:
        std::shared_ptr<generic::detail::EventGenericThreadsImpl<TDev>> m_spEventImpl;
    };
    namespace traits
    {
        //#############################################################################
        //! The CPU device event device get trait specialization.
        template<typename TDev>
        struct GetDev<EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(EventGenericThreads<TDev> const& event) -> TDev
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CPU device event test trait specialization.
        template<typename TDev>
        struct IsComplete<EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            //! \return If the event is not waiting within a queue (not enqueued or already handled).
            ALPAKA_FN_HOST static auto isComplete(EventGenericThreads<TDev> const& event) -> bool
            {
                std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                return event.m_spEventImpl->isReady();
            }
        };

        //#############################################################################
        //! The CPU non-blocking device queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>& queueImpl,
                EventGenericThreads<TDev>& event) -> void
            {
#if(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                alpaka::ignore_unused(queueImpl);
#endif
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Copy the shared pointer of the event implementation.
                // This is forwarded to the lambda that is enqueued into the queue to ensure that the event
                // implementation is alive as long as it is enqueued.
                auto spEventImpl(event.m_spEventImpl);

                // Setting the event state and enqueuing it has to be atomic.
                std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                ++spEventImpl->m_enqueueCount;

// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                auto const enqueueCount = spEventImpl->m_enqueueCount;

                // Enqueue a task that only resets the events flag if it is completed.
                spEventImpl->m_future = queueImpl.m_workerThread.enqueueTask([spEventImpl, enqueueCount]() {
                    std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);

                    // Nothing to do if it has been re-enqueued to a later position in the queue.
                    if(enqueueCount == spEventImpl->m_enqueueCount)
                    {
                        spEventImpl->m_LastReadyEnqueueCount = spEventImpl->m_enqueueCount;
                    }
                });
#endif
            }
        };
        //#############################################################################
        //! The CPU non-blocking device queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueGenericThreadsNonBlocking<TDev>& queue,
                EventGenericThreads<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                alpaka::enqueue(*queue.m_spQueueImpl, event);
            }
        };
        //#############################################################################
        //! The CPU blocking device queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>& queueImpl,
                EventGenericThreads<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                std::promise<void> promise;
                {
                    std::lock_guard<std::mutex> lk(queueImpl.m_mutex);

                    queueImpl.m_bCurrentlyExecutingTask = true;

                    auto& eventImpl(*event.m_spEventImpl);

                    {
                        // Setting the event state and enqueuing it has to be atomic.
                        std::lock_guard<std::mutex> evLk(eventImpl.m_mutex);

                        ++eventImpl.m_enqueueCount;
                        // NOTE: Difference to non-blocking version: directly set the event state instead of enqueuing.
                        eventImpl.m_LastReadyEnqueueCount = eventImpl.m_enqueueCount;

                        eventImpl.m_future = promise.get_future();
                    }

                    queueImpl.m_bCurrentlyExecutingTask = false;
                }
                promise.set_value();
            }
        };
        //#############################################################################
        //! The CPU blocking device queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericThreadsBlocking<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueGenericThreadsBlocking<TDev>& queue,
                EventGenericThreads<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                alpaka::enqueue(*queue.m_spQueueImpl, event);
            }
        };
    } // namespace traits
    namespace traits
    {
        namespace generic
        {
            template<typename TDev>
            ALPAKA_FN_HOST auto currentThreadWaitForDevice(TDev const& dev) -> void
            {
                // Get all the queues on the device at the time of invocation.
                // All queues added afterwards are ignored.
                auto vQueues(dev.getAllQueues());
                // Furthermore there should not even be a chance to enqueue something between getting the queues and
                // adding our wait events!
                std::vector<EventGenericThreads<TDev>> vEvents;
                for(auto&& spQueue : vQueues)
                {
                    vEvents.emplace_back(dev);
                    spQueue->enqueue(vEvents.back());
                }

                // Now wait for all the events.
                for(auto&& event : vEvents)
                {
                    wait(event);
                }
            }
        } // namespace generic

        //#############################################################################
        //! The CPU device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        template<typename TDev>
        struct CurrentThreadWaitFor<EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventGenericThreads<TDev> const& event) -> void
            {
                wait(*event.m_spEventImpl);
            }
        };
        //#############################################################################
        //! The CPU device event implementation thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        //!
        //! NOTE: This method is for internal usage only.
        template<typename TDev>
        struct CurrentThreadWaitFor<alpaka::generic::detail::EventGenericThreadsImpl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                alpaka::generic::detail::EventGenericThreadsImpl<TDev> const& eventImpl) -> void
            {
                std::unique_lock<std::mutex> lk(eventImpl.m_mutex);

                auto const enqueueCount = eventImpl.m_enqueueCount;
                eventImpl.wait(enqueueCount, lk);
            }
        };
        //#############################################################################
        //! The CPU non-blocking device queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<
            alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>,
            EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>& queueImpl,
#else
                alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>&,
#endif
                EventGenericThreads<TDev> const& event) -> void
            {
                // Copy the shared pointer of the event implementation.
                // This is forwarded to the lambda that is enqueued into the queue to ensure that the event
                // implementation is alive as long as it is enqueued.
                auto spEventImpl(event.m_spEventImpl);

                std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                if(!spEventImpl->isReady())
                {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    auto const enqueueCount = spEventImpl->m_enqueueCount;

                    // Enqueue a task that waits for the given event.
                    queueImpl.m_workerThread.enqueueTask([spEventImpl, enqueueCount]() {
                        std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
                        spEventImpl->wait(enqueueCount, lk2);
                    });
#endif
                }
            }
        };
        //#############################################################################
        //! The CPU non-blocking device queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericThreadsNonBlocking<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueGenericThreadsNonBlocking<TDev>& queue,
                EventGenericThreads<TDev> const& event) -> void
            {
                wait(*queue.m_spQueueImpl, event);
            }
        };
        //#############################################################################
        //! The CPU blocking device queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
                alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>& queueImpl,
                EventGenericThreads<TDev> const& event) -> void
            {
                alpaka::ignore_unused(queueImpl);

                // NOTE: Difference to non-blocking version: directly wait for event.
                wait(*event.m_spEventImpl);
            }
        };
        //#############################################################################
        //! The CPU blocking device queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericThreadsBlocking<TDev>, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueGenericThreadsBlocking<TDev>& queue,
                EventGenericThreads<TDev> const& event) -> void
            {
                wait(*queue.m_spQueueImpl, event);
            }
        };
        //#############################################################################
        //! The CPU non-blocking device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning
        //! execution.
        template<typename TDev>
        struct WaiterWaitFor<TDev, EventGenericThreads<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(TDev& dev, EventGenericThreads<TDev> const& event) -> void
            {
                // Get all the queues on the device at the time of invocation.
                // All queues added afterwards are ignored.
                auto vspQueues(dev.getAllQueues());

                // Let all the queues wait for this event.
                // Furthermore there should not even be a chance to enqueue something between getting the queues and
                // adding our wait events!
                for(auto&& spQueue : vspQueues)
                {
                    spQueue->wait(event);
                }
            }
        };

        //#############################################################################
        //! The CPU non-blocking device queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<typename TDev>
        struct CurrentThreadWaitFor<QueueGenericThreadsNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueGenericThreadsNonBlocking<TDev> const& queue) -> void
            {
                EventGenericThreads<TDev> event(getDev(queue));
                alpaka::enqueue(const_cast<QueueGenericThreadsNonBlocking<TDev>&>(queue), event);
                wait(event);
            }
        };
    } // namespace traits
} // namespace alpaka
