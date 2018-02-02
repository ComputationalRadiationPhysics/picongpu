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

#include <boost/core/ignore_unused.hpp>

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
                class StreamCpuSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCpuSyncImpl(
                        dev::DevCpu const & dev) :
                            m_dev(dev)
                    {}
                    //-----------------------------------------------------------------------------
                    StreamCpuSyncImpl(StreamCpuSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    StreamCpuSyncImpl(StreamCpuSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCpuSyncImpl const &) -> StreamCpuSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(StreamCpuSyncImpl &&) -> StreamCpuSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~StreamCpuSyncImpl() = default;

                public:
                    dev::DevCpu const m_dev;            //!< The device this stream is bound to.
                };
            }
        }

        //#############################################################################
        //! The CPU device stream.
        class StreamCpuSync final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCpuSync(
                dev::DevCpu const & dev) :
                    m_spStreamImpl(std::make_shared<cpu::detail::StreamCpuSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            StreamCpuSync(StreamCpuSync const &) = default;
            //-----------------------------------------------------------------------------
            StreamCpuSync(StreamCpuSync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCpuSync const &) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(StreamCpuSync &&) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamCpuSync const & rhs) const
            -> bool
            {
                return (m_spStreamImpl == rhs.m_spStreamImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamCpuSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~StreamCpuSync() = default;

        public:
            std::shared_ptr<cpu::detail::StreamCpuSyncImpl> m_spStreamImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU sync device stream device type trait specialization.
            template<>
            struct DevType<
                stream::StreamCpuSync>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU sync device stream device get trait specialization.
            template<>
            struct GetDev<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamCpuSync const & stream)
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
            //! The CPU sync device stream event type trait specialization.
            template<>
            struct EventType<
                stream::StreamCpuSync>
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
            //! The CPU sync device stream enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                stream::StreamCpuSync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuSync & stream,
                    TTask const & task)
                -> void
                {
                    boost::ignore_unused(stream);
                    task();
                }
            };
            //#############################################################################
            //! The CPU sync device stream test trait specialization.
            template<>
            struct Empty<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    stream::StreamCpuSync const & stream)
                -> bool
                {
                    boost::ignore_unused(stream);
                    return true;
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU sync device stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    stream::StreamCpuSync const & stream)
                -> void
                {
                    boost::ignore_unused(stream);
                }
            };
        }
    }
}
