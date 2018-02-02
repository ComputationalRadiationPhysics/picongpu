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
#include <alpaka/stream/Traits.hpp>
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
    namespace stream
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device stream implementation.
                class StreamCpuAsyncImpl final
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
                    ALPAKA_FN_HOST StreamCpuAsyncImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev),
                            m_workerThread(1u)
                    {}
                    //-----------------------------------------------------------------------------
                    StreamCpuAsyncImpl(StreamCpuAsyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    StreamCpuAsyncImpl(StreamCpuAsyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCpuAsyncImpl const &) -> StreamCpuAsyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCpuAsyncImpl &&) -> StreamCpuAsyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~StreamCpuAsyncImpl() noexcept(false)
                    {
                        m_dev.m_spDevCpuImpl->UnregisterAsyncStream(this);
                    }
                public:
                    dev::DevCpu const m_dev;            //!< The device this stream is bound to.

                    ThreadPool m_workerThread;
                };
            }
        }

        //#############################################################################
        //! The CPU device stream.
        class StreamCpuAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCpuAsync(
                dev::DevCpu const & dev) :
                    m_spStreamImpl(std::make_shared<cpu::detail::StreamCpuAsyncImpl>(dev))
            {
                dev.m_spDevCpuImpl->RegisterAsyncStream(m_spStreamImpl);
            }
            //-----------------------------------------------------------------------------
            StreamCpuAsync(StreamCpuAsync const &) = default;
            //-----------------------------------------------------------------------------
            StreamCpuAsync(StreamCpuAsync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCpuAsync const &) -> StreamCpuAsync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCpuAsync &&) -> StreamCpuAsync & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamCpuAsync const & rhs) const
            -> bool
            {
                return (m_spStreamImpl == rhs.m_spStreamImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamCpuAsync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~StreamCpuAsync() = default;

        public:
            std::shared_ptr<cpu::detail::StreamCpuAsyncImpl> m_spStreamImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU async device stream device type trait specialization.
            template<>
            struct DevType<
                stream::StreamCpuAsync>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU async device stream device get trait specialization.
            template<>
            struct GetDev<
                stream::StreamCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamCpuAsync const & stream)
                -> dev::DevCpu
                {
                    return stream.m_spStreamImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU async device stream event type trait specialization.
            template<>
            struct EventType<
                stream::StreamCpuAsync>
            {
                using type = event::EventCpu;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU async device stream enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                stream::StreamCpuAsync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    stream::StreamCpuAsync & stream,
                    TTask const & task)
#else
                    stream::StreamCpuAsync &,
                    TTask const &)
#endif
                -> void
                {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                    stream.m_spStreamImpl->m_workerThread.enqueueTask(
                        task);
#endif
                }
            };
            //#############################################################################
            //! The CPU async device stream test trait specialization.
            template<>
            struct Empty<
                stream::StreamCpuAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    stream::StreamCpuAsync const & stream)
                -> bool
                {
                    return stream.m_spStreamImpl->m_workerThread.isQueueEmpty();
                }
            };
        }
    }
}
