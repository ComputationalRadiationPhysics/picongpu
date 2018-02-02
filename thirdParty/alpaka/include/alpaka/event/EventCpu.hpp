/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/stream/StreamCpuAsync.hpp>
#include <alpaka/stream/StreamCpuSync.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/dev/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

#include <cassert>
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
                    ALPAKA_FN_HOST EventCpuImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev),
                            m_mutex(),
                            m_enqueueCount(0u),
                            m_LastReadyEnqueueCount(0u)
                    {}
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(EventCpuImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventCpuImpl(EventCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCpuImpl const &) -> EventCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCpuImpl &&) -> EventCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~EventCpuImpl() noexcept = default;

                    //-----------------------------------------------------------------------------
                    auto isReady() noexcept -> bool
                    {
                        return (m_LastReadyEnqueueCount == m_enqueueCount);
                    }

                    //-----------------------------------------------------------------------------
                    auto hasBeenReadieadSince(const std::size_t & enqueueCount) noexcept -> bool
                    {
                        return (m_LastReadyEnqueueCount >= enqueueCount);
                    }

                public:
                    dev::DevCpu const m_dev;                                //!< The device this event is bound to.

                    std::mutex mutable m_mutex;                             //!< The mutex used to synchronize access to the event.
                    std::condition_variable mutable m_conditionVariable;    //!< The condition signaling the event completion.
                    std::size_t m_enqueueCount;                             //!< The number of times this event has been enqueued.
                    std::size_t m_LastReadyEnqueueCount;                    //!< The time this event has been ready the last time.
                                                                            //!< Ready means that the event was not waiting within a stream (not enqueued or already completed).
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
            ALPAKA_FN_HOST EventCpu(
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
            ALPAKA_FN_HOST auto operator==(EventCpu const & rhs) const
            -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCpu const & rhs) const
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
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
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
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU async device stream enqueue trait specialization.
            template<>
            struct Enqueue<
                std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> & spStreamImpl,
#else
                    std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> &,
#endif
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                    ++spEventImpl->m_enqueueCount;

// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    auto const enqueueCount = spEventImpl->m_enqueueCount;

                    // Enqueue a task that only resets the events flag if it is completed.
                    spStreamImpl->m_workerThread.enqueueTask(
                        [spEventImpl, enqueueCount]()
                        {
                            std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);

                            // Nothing to do if it has been re-enqueued to a later position in the queue.
                            if(enqueueCount == spEventImpl->m_enqueueCount)
                            {
                                spEventImpl->m_LastReadyEnqueueCount = spEventImpl->m_enqueueCount;
                                lk2.unlock();
                                spEventImpl->m_conditionVariable.notify_all();
                            }
                        });
#endif
                }
            };
            //#############################################################################
            //! The CPU async device stream enqueue trait specialization.
            template<>
            struct Enqueue<
                stream::StreamCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuAsync & stream,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    stream::enqueue(stream.m_spStreamImpl, event);
                }
            };
            //#############################################################################
            //! The CPU sync device stream enqueue trait specialization.
            template<>
            struct Enqueue<
                std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl> & spStreamImpl,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    boost::ignore_unused(spStreamImpl);

                    auto spEventImpl(event.m_spEventImpl);

                    {
                        // Setting the event state and enqueuing it has to be atomic.
                        std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                        ++spEventImpl->m_enqueueCount;
                        // NOTE: Difference to async version: directly set the event state instead of enqueuing.
                        spEventImpl->m_LastReadyEnqueueCount = spEventImpl->m_enqueueCount;
                    }
                    spEventImpl->m_conditionVariable.notify_all();
                }
            };
            //#############################################################################
            //! The CPU sync device stream enqueue trait specialization.
            template<>
            struct Enqueue<
                stream::StreamCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuSync & stream,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    stream::enqueue(stream.m_spStreamImpl, event);
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
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
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
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
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

                    if(!spEventImpl->isReady())
                    {
                        auto const enqueueCount = spEventImpl->m_enqueueCount;
                        spEventImpl->m_conditionVariable.wait(
                            lk,
                            [spEventImpl, enqueueCount]{return spEventImpl->hasBeenReadieadSince(enqueueCount);});
                    }
                }
            };
            //#############################################################################
            //! The CPU async device stream event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> & spStreamImpl,
#else
                    std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> &,
#endif
                    event::EventCpu const & event)
                -> void
                {
                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                    if(!spEventImpl->isReady())
                    {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                        auto const enqueueCount = spEventImpl->m_enqueueCount;

                        // Enqueue a task that waits for the given event.
                        spStreamImpl->m_workerThread.enqueueTask(
                            [spEventImpl, enqueueCount]()
                            {
                                std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);

                                if(!spEventImpl->hasBeenReadieadSince(enqueueCount))
                                {
                                    spEventImpl->m_conditionVariable.wait(
                                        lk2,
                                        [spEventImpl, enqueueCount]{return spEventImpl->hasBeenReadieadSince(enqueueCount);});
                                }
                            });
#endif
                    }
                }
            };
            //#############################################################################
            //! The CPU async device stream event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCpuAsync & stream,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(stream.m_spStreamImpl, event);
                }
            };
            //#############################################################################
            //! The CPU sync device stream event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl> & spStreamImpl,
                    event::EventCpu const & event)
                -> void
                {
                    boost::ignore_unused(spStreamImpl);

                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);
                    // NOTE: Difference to async version: directly wait for event.
                    wait::wait(spEventImpl);
                }
            };
            //#############################################################################
            //! The CPU sync device stream event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCpuSync & stream,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(stream.m_spStreamImpl, event);
                }
            };
            //#############################################################################
            //! The CPU async device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
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
                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetAllAsyncStreamImpls());

                    // Let all the streams wait for this event.
                    // \TODO: This should be done atomically for all streams.
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    for(auto && spStream : vspStreams)
                    {
                        wait::wait(spStream, event);
                    }
                }
            };

            //#############################################################################
            //! The CPU async device stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    stream::StreamCpuAsync const & stream)
                -> void
                {
                    event::EventCpu event(
                        dev::getDev(stream));
                    stream::enqueue(
                        const_cast<stream::StreamCpuAsync &>(stream),
                        event);
                    wait::wait(
                        event);
                }
            };
        }
    }
}
