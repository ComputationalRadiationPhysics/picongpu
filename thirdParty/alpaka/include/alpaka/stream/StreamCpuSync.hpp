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

#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu

#include <alpaka/dev/Traits.hpp>                // dev::GetDev, dev::DevType
#include <alpaka/event/Traits.hpp>              // event::EventType
#include <alpaka/stream/Traits.hpp>             // stream::traits::Enqueue, ...
#include <alpaka/wait/Traits.hpp>               // CurrentThreadWaitFor, WaiterWaitFor

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/uuid/uuid.hpp>                  // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>       // boost::uuids::random_generator

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
                //#############################################################################
                class StreamCpuSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCpuSyncImpl(
                        dev::DevCpu const & dev) :
                            m_uuid(boost::uuids::random_generator()()),
                            m_dev(dev)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCpuSyncImpl(StreamCpuSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCpuSyncImpl(StreamCpuSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamCpuSyncImpl const &) -> StreamCpuSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamCpuSyncImpl &&) -> StreamCpuSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~StreamCpuSyncImpl() = default;

                public:
                    boost::uuids::uuid const m_uuid;    //!< The unique ID.
                    dev::DevCpu const m_dev;            //!< The device this stream is bound to.
                };
            }
        }

        //#############################################################################
        //! The CPU device stream.
        //#############################################################################
        class StreamCpuSync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCpuSync(
                dev::DevCpu const & dev) :
                    m_spSyncStreamCpu(std::make_shared<cpu::detail::StreamCpuSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCpuSync(StreamCpuSync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCpuSync(StreamCpuSync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamCpuSync const &) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamCpuSync &&) -> StreamCpuSync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamCpuSync const & rhs) const
            -> bool
            {
                return (m_spSyncStreamCpu->m_uuid == rhs.m_spSyncStreamCpu->m_uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamCpuSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~StreamCpuSync() = default;

        public:
            std::shared_ptr<cpu::detail::StreamCpuSyncImpl> m_spSyncStreamCpu;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU sync device stream device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                stream::StreamCpuSync>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU sync device stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamCpuSync const & stream)
                -> dev::DevCpu
                {
                    return stream.m_spSyncStreamCpu->m_dev;
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
            //#############################################################################
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
            //#############################################################################
            template<
                typename TTask>
            struct Enqueue<
                stream::StreamCpuSync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCpuSync & stream,
                    TTask & task)
                -> void
                {
                    boost::ignore_unused(stream);
                    task();
                }
                //-----------------------------------------------------------------------------
                //
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
            //#############################################################################
            template<>
            struct Empty<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                //
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
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCpuSync>
            {
                //-----------------------------------------------------------------------------
                //
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
