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
#include <alpaka/queue/QueueCpuAsync.hpp>
#include <alpaka/queue/QueueCpuSync.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/dev/Traits.hpp>

#include <mutex>
#include <condition_variable>
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
                class EventCpuImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(
                        dev::DevCpu const & dev) :
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
                    auto wait(std::size_t const & enqueueCount, std::unique_lock<std::mutex>& lk) noexcept -> void
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
        class EventCpu final
        {
        public:
            //-----------------------------------------------------------------------------
            EventCpu(
                dev::DevCpu const & dev) :
                    m_spEventImpl(std::make_shared<cpu::detail::EventCpuImpl>(dev))
            {}
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
            //! The CPU async device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl> & spQueueImpl,
#else
                    std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl> &,
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
                    spEventImpl->m_future = spQueueImpl->m_workerThread.enqueueTask(
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
            //! The CPU async device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuAsync & queue,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    queue::enqueue(queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU sync device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                std::shared_ptr<queue::cpu::detail::QueueCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    std::shared_ptr<queue::cpu::detail::QueueCpuSyncImpl> & spQueueImpl,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    alpaka::ignore_unused(spQueueImpl);

                    auto spEventImpl(event.m_spEventImpl);

                    std::promise<void> promise;
                    {
                        // Setting the event state and enqueuing it has to be atomic.
                        std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                        ++spEventImpl->m_enqueueCount;
                        // NOTE: Difference to async version: directly set the event state instead of enqueuing.
                        spEventImpl->m_LastReadyEnqueueCount = spEventImpl->m_enqueueCount;

                        spEventImpl->m_future = promise.get_future();
                    }
                    promise.set_value();
                }
            };
            //#############################################################################
            //! The CPU sync device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuSync & queue,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    queue::enqueue(queue.m_spQueueImpl, event);
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
                    wait::wait(event.m_spEventImpl);
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
                std::shared_ptr<event::cpu::detail::EventCpuImpl>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    std::shared_ptr<event::cpu::detail::EventCpuImpl> const & spEventImpl)
                -> void
                {
                    std::unique_lock<std::mutex> lk(spEventImpl->m_mutex);

                    auto const enqueueCount = spEventImpl->m_enqueueCount;
                    spEventImpl->wait(enqueueCount, lk);
                }
            };
            //#############################################################################
            //! The CPU async device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl> & spQueueImpl,
#else
                    std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl> &,
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
                        spQueueImpl->m_workerThread.enqueueTask(
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
            //! The CPU async device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCpuAsync & queue,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU sync device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<queue::cpu::detail::QueueCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    std::shared_ptr<queue::cpu::detail::QueueCpuSyncImpl> & spQueueImpl,
                    event::EventCpu const & event)
                -> void
                {
                    alpaka::ignore_unused(spQueueImpl);

                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the queue to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);
                    // NOTE: Difference to async version: directly wait for event.
                    wait::wait(spEventImpl);
                }
            };
            //#############################################################################
            //! The CPU sync device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCpuSync & queue,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(queue.m_spQueueImpl, event);
                }
            };
            //#############################################################################
            //! The CPU async device event wait trait specialization.
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
                        dev.m_spDevCpuImpl->GetAllAsyncQueueImpls());

                    // Let all the queues wait for this event.
                    // \TODO: This should be done atomically for all queues.
                    // Furthermore there should not even be a chance to enqueue something between getting the queues and adding our wait events!
                    for(auto && spQueue : vspQueues)
                    {
                        wait::wait(spQueue, event);
                    }
                }
            };

            //#############################################################################
            //! The CPU async device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuAsync const & queue)
                -> void
                {
                    event::EventCpu event(
                        dev::getDev(queue));
                    queue::enqueue(
                        const_cast<queue::QueueCpuAsync &>(queue),
                        event);
                    wait::wait(
                        event);
                }
            };
        }
    }
}
