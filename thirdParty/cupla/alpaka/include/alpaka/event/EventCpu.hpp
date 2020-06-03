/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/QueueCpuNonBlocking.hpp>
#include <alpaka/queue/QueueCpuBlocking.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/dev/Traits.hpp>

#include <mutex>
#include <condition_variable>
#include <future>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace event
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device event implementation.
                class EventCpuImpl final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, EventCpuImpl>
                {
                public:
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(
                        dev::DevCpu const & dev) noexcept :
                            m_dev(dev),
                            m_mutex(),
                            m_enqueueCount(0u),
                            m_LastReadyEnqueueCount(0u)
                    {}
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(EventCpuImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(EventCpuImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCpuImpl const &) -> EventCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCpuImpl &&) -> EventCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ~EventCpuImpl() noexcept = default;

                    //-----------------------------------------------------------------------------
                    auto isReady() noexcept -> bool
                    {
                        return (m_LastReadyEnqueueCount == m_enqueueCount);
                    }

                    //-----------------------------------------------------------------------------
                    auto wait(std::size_t const & enqueueCount, std::unique_lock<std::mutex>& lk) const noexcept -> void
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
                    dev::DevCpu const m_dev;                                //!< The device this event is bound to.

                    std::mutex mutable m_mutex;                             //!< The mutex used to synchronize access to the event.
                    std::shared_future<void> m_future;                      //!< The future signaling the event completion.
                    std::size_t m_enqueueCount;                             //!< The number of times this event has been enqueued.
                    std::size_t m_LastReadyEnqueueCount;                    //!< The time this event has been ready the last time.
                                                                            //!< Ready means that the event was not waiting within a queue (not enqueued or already completed).
                                                                            //!< If m_enqueueCount == m_LastReadyEnqueueCount, the event is currently not enqueued
                };
            }
        }

        //#############################################################################
        //! The CPU device event.
        class EventCpu final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, EventCpu>
        {
        public:
            //-----------------------------------------------------------------------------
            //! \param bBusyWaiting Unused. EventCpu never does busy waiting.
            EventCpu(
                dev::DevCpu const & dev,
                bool bBusyWaiting = true) :
                    m_spEventImpl(std::make_shared<cpu::detail::EventCpuImpl>(dev))
            { 
                alpaka::ignore_unused(bBusyWaiting);
            }
            //-----------------------------------------------------------------------------
            EventCpu(EventCpu const &) = default;
            //-----------------------------------------------------------------------------
            EventCpu(EventCpu &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCpu const &) -> EventCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCpu &&) -> EventCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator==(EventCpu const & rhs) const
            -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(EventCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventCpu() = default;

        public:
            std::shared_ptr<cpu::detail::EventCpuImpl> m_spEventImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCpu const & event)
                -> dev::DevCpu
                {
                    return event.m_spEventImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event test trait specialization.
            template<>
            struct Test<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a queue (not enqueued or already handled).
                ALPAKA_FN_HOST static auto test(
                    event::EventCpu const & event)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                    return event.m_spEventImpl->isReady();
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU non-blocking device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::cpu::detail::QueueCpuNonBlockingImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue::cpu::detail::QueueCpuNonBlockingImpl & queueImpl,
#else
                    queue::cpu::detail::QueueCpuNonBlockingImpl &,
#endif
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the queue to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                    ++spEventImpl->m_enqueueCount;

// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    auto const enqueueCount = spEventImpl->m_enqueueCount;

                    // Enqueue a task that only resets the events flag if it is completed.
                    spEventImpl->m_future = queueImpl.m_workerThread.enqueueTask(
                        [spEventImpl, enqueueCount]()
                        {
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
            template<>
            struct Enqueue<
                queue::QueueCpuNonBlocking,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuNonBlocking & queue,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    queue::enqueue(*queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::cpu::detail::QueueCpuBlockingImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::cpu::detail::QueueCpuBlockingImpl & queueImpl,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    std::promise<void> promise;
                    {
                        std::lock_guard<std::mutex> lk(queueImpl.m_mutex);

                        queueImpl.m_bCurrentlyExecutingTask = true;

                        auto & eventImpl(*event.m_spEventImpl);

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
            template<>
            struct Enqueue<
                queue::QueueCpuBlocking,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuBlocking & queue,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    queue::enqueue(*queue.m_spQueueImpl, event);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(*event.m_spEventImpl);
                }
            };
            //#############################################################################
            //! The CPU device event implementation thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            //!
            //! NOTE: This method is for internal usage only.
            template<>
            struct CurrentThreadWaitFor<
                event::cpu::detail::EventCpuImpl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::cpu::detail::EventCpuImpl const & eventImpl)
                -> void
                {
                    std::unique_lock<std::mutex> lk(eventImpl.m_mutex);

                    auto const enqueueCount = eventImpl.m_enqueueCount;
                    eventImpl.wait(enqueueCount, lk);
                }
            };
            //#############################################################################
            //! The CPU non-blocking device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::cpu::detail::QueueCpuNonBlockingImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue::cpu::detail::QueueCpuNonBlockingImpl & queueImpl,
#else
                    queue::cpu::detail::QueueCpuNonBlockingImpl &,
#endif
                    event::EventCpu const & event)
                -> void
                {
                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the queue to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                    if(!spEventImpl->isReady())
                    {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                        auto const enqueueCount = spEventImpl->m_enqueueCount;

                        // Enqueue a task that waits for the given event.
                        queueImpl.m_workerThread.enqueueTask(
                            [spEventImpl, enqueueCount]()
                            {
                                std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
                                spEventImpl->wait(enqueueCount, lk2);
                            });
#endif
                    }
                }
            };
            //#############################################################################
            //! The CPU non-blocking device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCpuNonBlocking,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCpuNonBlocking & queue,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(*queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU blocking device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::cpu::detail::QueueCpuBlockingImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::cpu::detail::QueueCpuBlockingImpl & queueImpl,
                    event::EventCpu const & event)
                -> void
                {
                    alpaka::ignore_unused(queueImpl);

                    // NOTE: Difference to non-blocking version: directly wait for event.
                    wait::wait(*event.m_spEventImpl);
                }
            };
            //#############################################################################
            //! The CPU blocking device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCpuBlocking,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCpuBlocking & queue,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(*queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU non-blocking device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevCpu,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCpu & dev,
                    event::EventCpu const & event)
                -> void
                {
                    // Get all the queues on the device at the time of invocation.
                    // All queues added afterwards are ignored.
                    auto vspQueues(
                        dev.m_spDevCpuImpl->GetAllQueues());

                    // Let all the queues wait for this event.
                    // Furthermore there should not even be a chance to enqueue something between getting the queues and adding our wait events!
                    for(auto && spQueue : vspQueues)
                    {
                        spQueue->wait(event);
                    }
                }
            };

            //#############################################################################
            //! The CPU non-blocking device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuNonBlocking const & queue)
                -> void
                {
                    event::EventCpu event(
                        dev::getDev(queue));
                    queue::enqueue(
                        const_cast<queue::QueueCpuNonBlocking &>(queue),
                        event);
                    wait::wait(
                        event);
                }
            };
        }
    }
}
