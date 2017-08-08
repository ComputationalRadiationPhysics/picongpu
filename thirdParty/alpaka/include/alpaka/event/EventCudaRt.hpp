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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>               // ALPAKA_FN_*, BOOST_LANG_CUDA

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/DevCudaRt.hpp>             // dev::DevCudaRt
#include <alpaka/dev/Traits.hpp>                // GetDev
#include <alpaka/event/Traits.hpp>              // event::traits::Test, ...
#include <alpaka/wait/Traits.hpp>               // CurrentThreadWaitFor

#include <alpaka/stream/StreamCudaRtAsync.hpp>  // stream::StreamCudaRtAsync
#include <alpaka/stream/StreamCudaRtSync.hpp>   // stream::StreamCudaRtSync
#include <alpaka/core/Cuda.hpp>                 // ALPAKA_CUDA_RT_CHECK

#include <stdexcept>                            // std::runtime_error
#include <memory>                               // std::shared_ptr
#include <functional>                           // std::bind

namespace alpaka
{
    namespace event
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT device event implementation.
                //#############################################################################
                class EventCudaImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaImpl(
                        dev::DevCudaRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_CudaEvent()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // Create the event on the current device with the specified flags. Valid flags include:
                        // - cudaEventDefault: Default event creation flag.
                        // - cudaEventBlockingSync : Specifies that event should use blocking synchronization.
                        //   A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                        // - cudaEventDisableTiming : Specifies that the created event does not need to record timing data.
                        //   Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                        ALPAKA_CUDA_RT_CHECK(
                            cudaEventCreateWithFlags(
                                &m_CudaEvent,
                                (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaImpl(EventCudaImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaImpl(EventCudaImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCudaImpl const &) -> EventCudaImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCudaImpl &&) -> EventCudaImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCudaImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaEventDestroy required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_dev.m_iDevice));
                        // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately
                        // and the resources associated with event will be released automatically once the device has completed event.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaEventDestroy(
                            m_CudaEvent));
                    }

                public:
                    dev::DevCudaRt const m_dev;   //!< The device this event is bound to.
                    cudaEvent_t m_CudaEvent;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT device event.
        //#############################################################################
        class EventCudaRt final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaRt(
                dev::DevCudaRt const & dev,
                bool bBusyWait = true) :
                    m_spEventCudaImpl(std::make_shared<cuda::detail::EventCudaImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaRt(EventCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaRt(EventCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCudaRt const &) -> EventCudaRt & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCudaRt &&) -> EventCudaRt & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCudaRt const & rhs) const
            -> bool
            {
                return (m_spEventCudaImpl->m_CudaEvent == rhs.m_spEventCudaImpl->m_CudaEvent);
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCudaRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~EventCudaRt() = default;

        public:
            std::shared_ptr<cuda::detail::EventCudaImpl> m_spEventCudaImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCudaRt const & event)
                -> dev::DevCudaRt
                {
                    return event.m_spEventCudaImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                event::EventCudaRt>
            {
                using type = event::EventCudaRt;
            };
            //#############################################################################
            //! The CPU device event create trait specialization.
            //#############################################################################
            template<>
            struct Create<
                event::EventCudaRt,
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto create(
                    dev::DevCudaRt const & dev)
                -> event::EventCudaRt
                {
                    return event::EventCudaRt(dev);
                }
            };
            //#############################################################################
            //! The CUDA RT device event test trait specialization.
            //#############################################################################
            template<>
            struct Test<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventCudaRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaEventQuery(
                            event.m_spEventCudaImpl->m_CudaEvent),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventCudaImpl->m_CudaEvent,
                        stream.m_spStreamCudaRtAsyncImpl->m_CudaStream));
                }
            };
            //#############################################################################
            //! The CUDA RT stream enqueue trait specialization.
            //#############################################################################
            template<>
            struct Enqueue<
                stream::StreamCudaRtSync,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventCudaImpl->m_CudaEvent,
                        stream.m_spStreamCudaRtSyncImpl->m_CudaStream));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        event.m_spEventCudaImpl->m_CudaEvent));
                }
            };
            //#############################################################################
            //! The CUDA RT stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCudaRtAsync,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCudaRtAsync & stream,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        stream.m_spStreamCudaRtAsyncImpl->m_CudaStream,
                        event.m_spEventCudaImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT stream event wait trait specialization.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                stream::StreamCudaRtSync,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    stream::StreamCudaRtSync & stream,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        stream.m_spStreamCudaRtSyncImpl->m_CudaStream,
                        event.m_spEventCudaImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT device event wait trait specialization.
            //!
            //! Any future work submitted in any stream of this device will wait for event to complete before beginning execution.
            //#############################################################################
            template<>
            struct WaiterWaitFor<
                dev::DevCudaRt,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaRt & dev,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        0,
                        event.m_spEventCudaImpl->m_CudaEvent,
                        0));
                }
            };
        }
    }
}

#endif
