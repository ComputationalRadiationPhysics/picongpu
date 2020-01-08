/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/cpu/ICpuQueue.hpp>

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
#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif
                //#############################################################################
                //! The CPU device queue implementation.
                class QueueCpuNonBlockingImpl final : public cpu::ICpuQueue
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
                    QueueCpuNonBlockingImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev),
                            m_workerThread(1u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuNonBlockingImpl(QueueCpuNonBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuNonBlockingImpl(QueueCpuNonBlockingImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuNonBlockingImpl const &) -> QueueCpuNonBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuNonBlockingImpl &&) -> QueueCpuNonBlockingImpl & = delete;

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

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device queue.
        class QueueCpuNonBlocking final
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuNonBlocking(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuNonBlockingImpl>(dev))
            {
                dev.m_spDevCpuImpl->RegisterQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueCpuNonBlocking(QueueCpuNonBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuNonBlocking(QueueCpuNonBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuNonBlocking const &) -> QueueCpuNonBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuNonBlocking &&) -> QueueCpuNonBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuNonBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuNonBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuNonBlocking() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuNonBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU non-blocking device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuNonBlocking>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU non-blocking device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuNonBlocking const & queue)
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
            //! The CPU non-blocking device queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCpuNonBlocking>
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
            //! The CPU non-blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCpuNonBlocking,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    queue::QueueCpuNonBlocking & queue,
                    TTask const & task)
#else
                    queue::QueueCpuNonBlocking &,
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
            //! The CPU non-blocking device queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCpuNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuNonBlocking const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_workerThread.isIdle();
                }
            };
        }
    }
}

#include <alpaka/event/EventCpu.hpp>
