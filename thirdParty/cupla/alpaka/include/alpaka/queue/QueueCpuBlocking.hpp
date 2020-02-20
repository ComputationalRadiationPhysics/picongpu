/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/cpu/ICpuQueue.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <atomic>
#include <mutex>

namespace alpaka
{
    namespace event
    {
        class EventCpu;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cpu
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
                class QueueCpuBlockingImpl final : public cpu::ICpuQueue
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                public:
                    //-----------------------------------------------------------------------------
                    QueueCpuBlockingImpl(
                        dev::DevCpu const & dev) noexcept :
                            m_dev(dev),
                            m_bCurrentlyExecutingTask(false)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuBlockingImpl(QueueCpuBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuBlockingImpl(QueueCpuBlockingImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuBlockingImpl const &) -> QueueCpuBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuBlockingImpl &&) -> QueueCpuBlockingImpl & = delete;

                    //-----------------------------------------------------------------------------
                    void enqueue(event::EventCpu & ev) final
                    {
                        queue::enqueue(*this, ev);
                    }

                    //-----------------------------------------------------------------------------
                    void wait(event::EventCpu const & ev) final
                    {
                        wait::wait(*this, ev);
                    }

                public:
                    dev::DevCpu const m_dev;            //!< The device this queue is bound to.
                    std::mutex mutable m_mutex;
                    std::atomic<bool> m_bCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        class QueueCpuBlocking final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueCpuBlocking>
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuBlocking(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuBlockingImpl>(dev))
            {
                dev.m_spDevCpuImpl->RegisterQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueCpuBlocking(QueueCpuBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuBlocking(QueueCpuBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuBlocking const &) -> QueueCpuBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuBlocking &&) -> QueueCpuBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuBlocking() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuBlocking>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU blocking device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuBlocking const & queue)
                -> dev::DevCpu
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
            template<>
            struct EventType<
                queue::QueueCpuBlocking>
            {
                using type = event::EventCpu;
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
                typename TTask>
            struct Enqueue<
                queue::QueueCpuBlocking,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuBlocking & queue,
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
            template<>
            struct Empty<
                queue::QueueCpuBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuBlocking const & queue)
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
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuBlocking const & queue)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                }
            };
        }
    }
}

#include <alpaka/event/EventCpu.hpp>
