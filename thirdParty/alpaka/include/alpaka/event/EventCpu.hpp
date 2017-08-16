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

#include <alpaka/dev/DevCpu.hpp>            // dev::DevCpu
#include <alpaka/stream/StreamCpuAsync.hpp> // stream::StreamCpuAsync
#include <alpaka/stream/StreamCpuSync.hpp>  // stream::StreamCpuSync

#include <alpaka/dev/Traits.hpp>            // GetDev
#include <alpaka/event/Traits.hpp>          // event::traits::Test, ...
#include <alpaka/wait/Traits.hpp>           // CurrentThreadWaitFor
#include <alpaka/dev/Traits.hpp>            // GetDev

#include <boost/uuid/uuid.hpp>              // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>   // boost::uuids::random_generator
#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <cassert>                          // assert
#include <mutex>                            // std::mutex
#include <condition_variable>               // std::condition_variable
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
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
                //#############################################################################
                class EventCpuImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuImpl(
                        dev::DevCpu const & dev) :
                            m_uuid(boost::uuids::random_generator()()),
                            m_dev(dev),
                            m_Mutex(),
                            m_canceledEnqueueCount(0),
                            m_bIsReady(true),
                            m_bIsWaitedFor(false)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuImpl(EventCpuImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuImpl(EventCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCpuImpl const &) -> EventCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCpuImpl &&) -> EventCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCpuImpl() noexcept
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    {
                        // If an event is enqueued to a stream and gets waited on but destructed before it is completed it is kept alive until completed.
                        // This can never happen.
                        assert(!m_bIsWaitedFor);
                    }
#else
                    = default;
#endif
                public:
                    boost::uuids::uuid const m_uuid;                        //!< The unique ID.
                    dev::DevCpu const m_dev;                                //!< The device this event is bound to.

                    std::mutex mutable m_Mutex;                             //!< The mutex used to synchronize access to the event.

                    std::condition_variable mutable m_ConditionVariable;    //!< The condition signaling the event completion.
                    std::size_t m_canceledEnqueueCount;                     //!< The number of successive re-enqueues while it was already in the queue. Reset on completion.
                    bool m_bIsReady;                                        //!< If the event is not waiting within a stream (not enqueued or already completed).

                    bool m_bIsWaitedFor;                                    //!< If a (one or multiple) streams wait for this event. The event can not be changed (deleted/re-enqueued) until completion.
                };
            }
        }

        //#############################################################################
        //! The CPU device event.
        //#############################################################################
        class EventCpu final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpu(
                dev::DevCpu const & dev) :
                    m_spEventCpuImpl(std::make_shared<cpu::detail::EventCpuImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpu(EventCpu const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpu(EventCpu &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCpu const &) -> EventCpu & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCpu &&) -> EventCpu & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCpu const & rhs) const
            -> bool
            {
                return (m_spEventCpuImpl->m_uuid == rhs.m_spEventCpuImpl->m_uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            std::shared_ptr<cpu::detail::EventCpuImpl> m_spEventCpuImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCpu const & event)
                -> dev::DevCpu
                {
                    return event.m_spEventCpuImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                event::EventCpu>
            {
                using type = event::EventCpu;
            };
            //#############################################################################
            //! The CPU device event create trait specialization.
            //#############################################################################
            template<>
            struct Create<
                event::EventCpu,
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto create(
                    dev::DevCpu const & dev)
                -> event::EventCpu
                {
                    return event::EventCpu(dev);
                }
            };
            //#############################################################################
            //! The CPU device event test trait specialization.
            //#############################################################################
            template<>
            struct Test<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventCpu const & event)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(event.m_spEventCpuImpl->m_Mutex);

                    return event.m_spEventCpuImpl->m_bIsReady;
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
            //#############################################################################
            template<>
            struct Enqueue<
                std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
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
                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventCpuImpl(event.m_spEventCpuImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);

                    // This is a invariant: If the event is ready (not enqueued) there can not be anybody waiting for it.
                    assert(!(spEventCpuImpl->m_bIsReady && spEventCpuImpl->m_bIsWaitedFor));

                    // If it is enqueued ...
                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        // ... and somebody is waiting for it, it can NOT be re-enqueued.
                        if(spEventCpuImpl->m_bIsWaitedFor)
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                            std::cout << BOOST_CURRENT_FUNCTION << "WARNING: The event to enqueue is already enqueued AND waited on. It can NOT be re-enqueued!" << std::endl;
#endif
                            return;
                        }
                        // ... and nobody is waiting for it, increment the cancel counter.
                        else
                        {
                            ++spEventCpuImpl->m_canceledEnqueueCount;
                        }
                    }
                    // If it is not enqueued, set its state to enqueued.
                    else
                    {
                        spEventCpuImpl->m_bIsReady = false;
                    }

                    // We can not unlock the mutex here, because the order of events enqueued has to be identical to the call order.
                    // Unlocking here would allow a later enqueue call to complete before this event is enqueued.
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    // Enqueue a task that only resets the events flag if it is completed.
                    spStreamImpl->m_workerThread.enqueueTask(
                        [spEventCpuImpl]()
                        {
                            {
                                std::lock_guard<std::mutex> lk2(spEventCpuImpl->m_Mutex);
                                // Nothing to do if it has been re-enqueued to a later position in the queue.
                                if(spEventCpuImpl->m_canceledEnqueueCount > 0)
                                {
                                    --spEventCpuImpl->m_canceledEnqueueCount;
                                    return;
                                }
                                else
                                {
                                    spEventCpuImpl->m_bIsWaitedFor = false;
                                    spEventCpuImpl->m_bIsReady = true;
                                }
                            }
                            spEventCpuImpl->m_ConditionVariable.notify_all();
                        });
#endif
                }
            };
            //#############################################################################
            //! The CPU async device stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuAsync & stream,
                    event::EventCpu & event)
                -> void
                {
                    stream::enqueue(stream.m_spAsyncStreamCpu, event);
                }
            };
            //#############################################################################
            //! The CPU sync device stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl> & spStreamImpl,
                    event::EventCpu & event)
                -> void
                {
                    boost::ignore_unused(spStreamImpl);

                    {
                        // Copy the shared pointer of the event implementation.
                        // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                        auto spEventCpuImpl(event.m_spEventCpuImpl);

                        // Setting the event state and enqueuing it has to be atomic.
                        std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);

                        // This is a invariant: If the event is ready (not enqueued) there can not be anybody waiting for it.
                        assert(!(spEventCpuImpl->m_bIsReady && spEventCpuImpl->m_bIsWaitedFor));

                        // If it is enqueued ...
                        if(!spEventCpuImpl->m_bIsReady)
                        {
                            // ... and somebody is waiting for it, it can NOT be re-enqueued.
                            if(spEventCpuImpl->m_bIsWaitedFor)
                            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                std::cout << BOOST_CURRENT_FUNCTION << "WARNING: The event to enqueue is already enqueued AND waited on. It can NOT be re-enqueued!" << std::endl;
#endif
                                return;
                            }
                            // ... and nobody is waiting for it, increment the cancel counter.
                            else
                            {
                                ++spEventCpuImpl->m_canceledEnqueueCount;
                            }
                        }
                        // If it is not enqueued, set its state to enqueued.
                        else
                        {
                            spEventCpuImpl->m_bIsReady = false;
                        }

                        // NOTE: Difference to async version: directly reset the event flag instead of enqueuing.

                        // Nothing to do if it has been re-enqueued to a later position in any queue.
                        if(spEventCpuImpl->m_canceledEnqueueCount > 0)
                        {
                            --spEventCpuImpl->m_canceledEnqueueCount;
                            return;
                        }
                        else
                        {
                            spEventCpuImpl->m_bIsWaitedFor = false;
                            spEventCpuImpl->m_bIsReady = true;
                        }
                    }
                    event.m_spEventCpuImpl->m_ConditionVariable.notify_all();
                }
            };
            //#############################################################################
            //! The CPU sync device stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuSync & stream,
                    event::EventCpu & event)
                -> void
                {
                    stream::enqueue(stream.m_spSyncStreamCpu, event);
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
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(event.m_spEventCpuImpl);
                }
            };
            //#############################################################################
            //! The CPU device event implementation thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //!
            //! NOTE: This method is for internal usage only.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                std::shared_ptr<event::cpu::detail::EventCpuImpl>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    std::shared_ptr<event::cpu::detail::EventCpuImpl> const & spEventCpuImpl)
                -> void
                {
                    std::unique_lock<std::mutex> lk(spEventCpuImpl->m_Mutex);

                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        spEventCpuImpl->m_bIsWaitedFor = true;
                        spEventCpuImpl->m_ConditionVariable.wait(
                            lk,
                            [spEventCpuImpl]{return spEventCpuImpl->m_bIsReady;});
                    }
                }
            };
            //#############################################################################
            //! The CPU async device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
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
                    auto spEventCpuImpl(event.m_spEventCpuImpl);

                    std::lock_guard<std::mutex> lk(spEventCpuImpl->m_Mutex);
                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        spEventCpuImpl->m_bIsWaitedFor = true;
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                        // Enqueue a task that waits for the given event.
                        spStreamImpl->m_workerThread.enqueueTask(
                            [spEventCpuImpl]()
                            {
                                wait::wait(spEventCpuImpl);
                            });
#endif
                    }
                }
            };
            //#############################################################################
            //! The CPU async device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuAsync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCpuAsync & stream,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(stream.m_spAsyncStreamCpu, event);
                }
            };
            //#############################################################################
            //! The CPU sync device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl>,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    std::shared_ptr<stream::cpu::detail::StreamCpuSyncImpl> & spStreamImpl,
                    event::EventCpu const & event)
                -> void
                {
                    boost::ignore_unused(spStreamImpl);

                    // Copy the shared pointer of the event implementation.
                    // This is forwarded to the lambda that is enqueued into the stream to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventCpuImpl(event.m_spEventCpuImpl);
                    // NOTE: Difference to async version: directly wait for event.
                    wait::wait(spEventCpuImpl);
                }
            };
            //#############################################################################
            //! The CPU sync device stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCpuSync,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCpuSync & stream,
                    event::EventCpu const & event)
                -> void
                {
                    wait::wait(stream.m_spSyncStreamCpu, event);
                }
            };
            //#############################################################################
            //! The CPU async device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                dev::DevCpu,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                //
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
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuAsync>
            {
                //-----------------------------------------------------------------------------
                //
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
