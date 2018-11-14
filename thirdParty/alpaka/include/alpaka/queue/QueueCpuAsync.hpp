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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/ConcurrentExecPool.hpp>

#include <type_traits>
#include <thread>
#include <mutex>
#include <future>

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
                class QueueCpuAsyncImpl final
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
                    QueueCpuAsyncImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev),
                            m_workerThread(1u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuAsyncImpl(QueueCpuAsyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuAsyncImpl(QueueCpuAsyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuAsyncImpl const &) -> QueueCpuAsyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuAsyncImpl &&) -> QueueCpuAsyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~QueueCpuAsyncImpl() = default;

                public:
                    dev::DevCpu const m_dev;            //!< The device this queue is bound to.

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        class QueueCpuAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuAsync(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuAsyncImpl>(dev))
            {
                dev.m_spDevCpuImpl->RegisterAsyncQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueCpuAsync(QueueCpuAsync const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuAsync(QueueCpuAsync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuAsync const &) -> QueueCpuAsync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuAsync &&) -> QueueCpuAsync & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuAsync const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuAsync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuAsync() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuAsyncImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU async device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuAsync>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU async device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuAsync const & queue)
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
            //! The CPU async device queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCpuAsync>
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
            //! The CPU async device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCpuAsync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue::QueueCpuAsync & queue,
                    TTask const & task)
#else
                    queue::QueueCpuAsync &,
                    TTask const &)
#endif
                -> void
                {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue.m_spQueueImpl->m_workerThread.enqueueTask(
                        task);
#endif
                }
            };
            //#############################################################################
            //! The CPU async device queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuAsync const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_workerThread.isIdle();
                }
            };
        }
    }
}
