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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, BOOST_LANG_CUDA

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/DevCudaRt.hpp>     // dev::DevCudaRt

#include <alpaka/dev/Traits.hpp>        // dev::GetDev, dev::DevType
#include <alpaka/event/Traits.hpp>      // event::EventType
#include <alpaka/stream/Traits.hpp>     // stream::traits::Enqueue, ...
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/Cuda.hpp>         // ALPAKA_CUDA_RT_CHECK

#include <stdexcept>                    // std::runtime_error
#include <memory>                       // std::shared_ptr
#include <functional>                   // std::bind

namespace alpaka
{
    namespace event
    {
        class EventCudaRt;
    }
}

namespace alpaka
{
    namespace stream
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT stream implementation.
                //#############################################################################
                class StreamCudaRtSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    StreamCudaRtSyncImpl(
                        dev::DevCudaRt const & dev) :
                            m_dev(dev),
                            m_CudaStream()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // - cudaStreamDefault: Default stream creation flag.
                        // - cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream),
                        //   and that the created stream should perform no implicit synchronization with stream 0.
                        // Create the stream on the current device.
                        // NOTE: cudaStreamNonBlocking is required to match the semantic implemented in the alpaka CPU stream.
                        // It would be too much work to implement implicit default stream synchronization on CPU.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamCreateWithFlags(
                                &m_CudaStream,
                                cudaStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCudaRtSyncImpl(StreamCudaRtSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST StreamCudaRtSyncImpl(StreamCudaRtSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamCudaRtSyncImpl const &) -> StreamCudaRtSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(StreamCudaRtSyncImpl &&) -> StreamCudaRtSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ~StreamCudaRtSyncImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaStreamDestroy required?
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the stream when cudaStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamDestroy(
                                m_CudaStream));
                    }

                public:
                    dev::DevCudaRt const m_dev;   //!< The device this stream is bound to.
                    cudaStream_t m_CudaStream;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT stream.
        //#############################################################################
        class StreamCudaRtSync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCudaRtSync(
                dev::DevCudaRt const & dev) :
                m_spStreamCudaRtSyncImpl(std::make_shared<cuda::detail::StreamCudaRtSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCudaRtSync(StreamCudaRtSync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST StreamCudaRtSync(StreamCudaRtSync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamCudaRtSync const &) -> StreamCudaRtSync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(StreamCudaRtSync &&) -> StreamCudaRtSync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(StreamCudaRtSync const & rhs) const
            -> bool
            {
                return (m_spStreamCudaRtSyncImpl->m_CudaStream == rhs.m_spStreamCudaRtSyncImpl->m_CudaStream);
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(StreamCudaRtSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~StreamCudaRtSync() = default;

        public:
            std::shared_ptr<cuda::detail::StreamCudaRtSyncImpl> m_spStreamCudaRtSyncImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                stream::StreamCudaRtSync>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The CUDA RT stream device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    stream::StreamCudaRtSync const & stream)
                -> dev::DevCudaRt
                {
                    return stream.m_spStreamCudaRtSyncImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                stream::StreamCudaRtSync>
            {
                using type = event::EventCudaRt;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream test trait specialization.
            //#############################################################################
            template<>
            struct Empty<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    stream::StreamCudaRtSync const & stream)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for streams on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaStreamQuery(
                            stream.m_spStreamCudaRtSyncImpl->m_CudaStream),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT stream thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the stream has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                stream::StreamCudaRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    stream::StreamCudaRtSync const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for streams on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaStreamSynchronize(
                        stream.m_spStreamCudaRtSyncImpl->m_CudaStream));
                }
            };
        }
    }
}

#endif
