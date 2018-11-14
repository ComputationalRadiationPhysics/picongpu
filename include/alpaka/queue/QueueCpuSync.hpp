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

#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

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
                //#############################################################################
                //! The CPU device queue implementation.
                class QueueCpuSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    QueueCpuSyncImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev),
                            m_bCurrentlyExecutingTask(false)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuSyncImpl(QueueCpuSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuSyncImpl(QueueCpuSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuSyncImpl const &) -> QueueCpuSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuSyncImpl &&) -> QueueCpuSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~QueueCpuSyncImpl() = default;

                public:
                    dev::DevCpu const m_dev;            //!< The device this queue is bound to.
                    bool m_bCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        class QueueCpuSync final
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuSync(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueCpuSync(QueueCpuSync const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuSync(QueueCpuSync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuSync const &) -> QueueCpuSync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuSync &&) -> QueueCpuSync & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuSync const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuSync() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuSyncImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU sync device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuSync>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU sync device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuSync const & queue)
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
            //! The CPU sync device queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCpuSync>
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
            //! The CPU sync device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCpuSync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuSync & queue,
                    TTask const & task)
                -> void
                {
                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;
                    task();
                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
                }
            };
            //#############################################################################
            //! The CPU sync device queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuSync const & queue)
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
            //! The CPU sync device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuSync const & queue)
                -> void
                {
                    alpaka::ignore_unused(queue);
                }
            };
        }
    }
}
