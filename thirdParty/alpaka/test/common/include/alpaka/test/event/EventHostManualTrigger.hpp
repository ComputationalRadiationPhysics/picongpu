/**
* \file
* Copyright 2017 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

#include <mutex>                        // std::mutex
#include <condition_variable>           // std::condition_variable

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    //-----------------------------------------------------------------------------
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test event specifics.
        //-----------------------------------------------------------------------------
        namespace event
        {
            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TDev>
                struct EventHostManualTriggerType;
            }

            //#############################################################################
            //! The event type trait alias template to remove the ::type.
            //#############################################################################
            template<
                typename TDev>
            using EventHostManualTrigger = typename traits::EventHostManualTriggerType<TDev>::type;

            namespace cpu
            {
                namespace detail
                {
                    //#############################################################################
                    //! Event that can be enqueued into a stream and can be triggered by the Host.
                    //#############################################################################
                    class EventHostManualTriggerCpuImpl
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(
                            dev::DevCpu const & dev) :
                                m_dev(dev),
                                m_mutex(),
                                m_enqueueCount(0u),
                                m_bIsReady(true)
                    {}
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl const & other) = delete;
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl &&) = default;
                        //-----------------------------------------------------------------------------
                        //! Copy assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator=(EventHostManualTriggerCpuImpl const &) -> EventHostManualTriggerCpuImpl & = delete;
                        //-----------------------------------------------------------------------------
                        //! Move assignment operator.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator=(EventHostManualTriggerCpuImpl &&) -> EventHostManualTriggerCpuImpl & = default;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        void trigger()
                        {
                            {
                                std::unique_lock<std::mutex> lock(m_mutex);
                                m_bIsReady = true;
                            }
                            m_conditionVariable.notify_one();
                        }

                    public:
                        dev::DevCpu const m_dev;                                //!< The device this event is bound to.

                        mutable std::mutex m_mutex;                             //!< The mutex used to synchronize access to the event.

                        mutable std::condition_variable m_conditionVariable;    //!< The condition signaling the event completion.
                        std::size_t m_enqueueCount;                             //!< The number of times this event has been enqueued.

                        bool m_bIsReady;                                        //!< If the event is not waiting within a stream (not enqueued or already completed).
                    };
                }
            }

            //#############################################################################
            //! Event that can be enqueued into a stream and can be triggered by the Host.
            //#############################################################################
            class EventHostManualTriggerCpu
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST EventHostManualTriggerCpu(
                    dev::DevCpu const & dev) :
                        m_spEventImpl(std::make_shared<cpu::detail::EventHostManualTriggerCpuImpl>(dev))
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST EventHostManualTriggerCpu(EventHostManualTriggerCpu const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST EventHostManualTriggerCpu(EventHostManualTriggerCpu &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(EventHostManualTriggerCpu const &) -> EventHostManualTriggerCpu & = default;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(EventHostManualTriggerCpu &&) -> EventHostManualTriggerCpu & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCpu const & rhs) const
                -> bool
                {
                    return (m_spEventImpl == rhs.m_spEventImpl);
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCpu const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }

                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                void trigger()
                {
                    m_spEventImpl->trigger();
                }

            public:
                std::shared_ptr<cpu::detail::EventHostManualTriggerCpuImpl> m_spEventImpl;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct EventHostManualTriggerType<
                    alpaka::dev::DevCpu>
                {
                    using type = alpaka::test::event::EventHostManualTriggerCpu;
                };
            }
        }
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
                test::event::EventHostManualTriggerCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    test::event::EventHostManualTriggerCpu const & event)
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
            //#############################################################################
            template<>
            struct Test<
                test::event::EventHostManualTriggerCpu>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    test::event::EventHostManualTriggerCpu const & event)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                    return event.m_spEventImpl->m_bIsReady;
                }
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCpuAsync,
                test::event::EventHostManualTriggerCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    stream::StreamCpuAsync & stream,
#else
                    stream::StreamCpuAsync &,
#endif
                    test::event::EventHostManualTriggerCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

                    // The event should not yet be enqueued.
                    assert(spEventImpl->m_bIsReady);

                    // Set its state to enqueued.
                    spEventImpl->m_bIsReady = false;

                    // Increment the enqueue counter. This is used to skip waits for events that had already been finished and re-enqueued which would lead to deadlocks.
                    ++spEventImpl->m_enqueueCount;

                    // Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    auto const enqueueCount = spEventImpl->m_enqueueCount;

                    // Enqueue a task that only resets the events flag if it is completed.
                    stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                        [spEventImpl, enqueueCount]()
                        {
                            std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
                            spEventImpl->m_conditionVariable.wait(
                                lk2,
                                [spEventImpl, enqueueCount]
                                {
                                    return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady;
                                });
                        });
#endif
                }
            };
            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCpuSync,
                test::event::EventHostManualTriggerCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuSync &,
                    test::event::EventHostManualTriggerCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
                    auto spEventImpl(event.m_spEventImpl);

                    // Setting the event state and enqueuing it has to be atomic.
                    std::unique_lock<std::mutex> lk(spEventImpl->m_mutex);

                    // The event should not yet be enqueued.
                    assert(spEventImpl->m_bIsReady);

                    // Set its state to enqueued.
                    spEventImpl->m_bIsReady = false;

                    // Increment the enqueue counter. This is used to skip waits for events that had already been finished and re-enqueued which would lead to deadlocks.
                    ++spEventImpl->m_enqueueCount;

                    auto const enqueueCount = spEventImpl->m_enqueueCount;

                    spEventImpl->m_conditionVariable.wait(
                        lk,
                        [spEventImpl, enqueueCount]
                        {
                            return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady;
                        });
                }
            };
        }
    }
}
