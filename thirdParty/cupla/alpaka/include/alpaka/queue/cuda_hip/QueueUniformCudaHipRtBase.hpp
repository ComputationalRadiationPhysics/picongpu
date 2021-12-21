/* Copyright 2020 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once


#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <memory>

namespace alpaka
{
    namespace uniform_cuda_hip
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA/HIP RT blocking queue implementation.
            class QueueUniformCudaHipRtImpl final
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt const& dev)
                    : m_dev(dev)
                    , m_UniformCudaHipQueue()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.m_iDevice));

                    // - [cuda/hip]StreamDefault: Default queue creation flag.
                    // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run
                    // concurrently with work in queue 0 (the NULL queue),
                    //   and that the created queue should perform no implicit synchronization with queue 0.
                    // Create the queue on the current device.
                    // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka
                    // CPU queue. It would be too much work to implement implicit default queue synchronization on CPU.

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                        StreamCreateWithFlags)(&m_UniformCudaHipQueue, ALPAKA_API_PREFIX(StreamNonBlocking)));
                }
                //-----------------------------------------------------------------------------
                QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl const&) = delete;
                //-----------------------------------------------------------------------------
                QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl&&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(QueueUniformCudaHipRtImpl const&) -> QueueUniformCudaHipRtImpl& = delete;
                //-----------------------------------------------------------------------------
                auto operator=(QueueUniformCudaHipRtImpl&&) -> QueueUniformCudaHipRtImpl& = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST ~QueueUniformCudaHipRtImpl()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device. \TODO: Is setting the current device before [cuda/hip]StreamDestroy
                    // required?

                    // In case the device is still doing work in the queue when [cuda/hip]StreamDestroy() is called,
                    // the function will return immediately and the resources associated with queue will be released
                    // automatically once the device has completed all work in queue.
                    // -> No need to synchronize here.

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.m_iDevice));
                    // In case the device is still doing work in the queue when cuda/hip-StreamDestroy() is called, the
                    // function will return immediately and the resources associated with queue will be released
                    // automatically once the device has completed all work in queue.
                    // -> No need to synchronize here.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamDestroy)(m_UniformCudaHipQueue));
                }

            public:
                DevUniformCudaHipRt const m_dev; //!< The device this queue is bound to.
                ALPAKA_API_PREFIX(Stream_t) m_UniformCudaHipQueue;
            };

            //#############################################################################
            //! The CUDA RT blocking queue.
            class QueueUniformCudaHipRtBase
                : public concepts::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRtBase>
                , public concepts::Implements<ConceptQueue, QueueUniformCudaHipRtBase>
                , public concepts::Implements<ConceptGetDev, QueueUniformCudaHipRtBase>
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST QueueUniformCudaHipRtBase(DevUniformCudaHipRt const& dev)
                    : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl>(dev))
                {
                }
                //-----------------------------------------------------------------------------
                QueueUniformCudaHipRtBase(QueueUniformCudaHipRtBase const&) = default;
                //-----------------------------------------------------------------------------
                QueueUniformCudaHipRtBase(QueueUniformCudaHipRtBase&&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(QueueUniformCudaHipRtBase const&) -> QueueUniformCudaHipRtBase& = default;
                //-----------------------------------------------------------------------------
                auto operator=(QueueUniformCudaHipRtBase&&) -> QueueUniformCudaHipRtBase& = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRtBase const& rhs) const -> bool
                {
                    return (m_spQueueImpl == rhs.m_spQueueImpl);
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRtBase const& rhs) const -> bool
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                ~QueueUniformCudaHipRtBase() = default;

            public:
                std::shared_ptr<QueueUniformCudaHipRtImpl> m_spQueueImpl;
            };
        } // namespace detail
    } // namespace uniform_cuda_hip

    namespace traits
    {
        //#############################################################################
        //! The CUDA/HIP RT non-blocking queue device get trait specialization.
        template<>
        struct GetDev<uniform_cuda_hip::detail::QueueUniformCudaHipRtBase>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(uniform_cuda_hip::detail::QueueUniformCudaHipRtBase const& queue)
                -> DevUniformCudaHipRt
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CUDA/HIP RT blocking queue test trait specialization.
        template<>
        struct Empty<uniform_cuda_hip::detail::QueueUniformCudaHipRtBase>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto empty(uniform_cuda_hip::detail::QueueUniformCudaHipRtBase const& queue) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for queues on non current device.
                ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = ALPAKA_API_PREFIX(StreamQuery)(queue.m_spQueueImpl->m_UniformCudaHipQueue),
                    ALPAKA_API_PREFIX(ErrorNotReady));
                return (ret == ALPAKA_API_PREFIX(Success));
            }
        };

        //#############################################################################
        //! The CUDA/HIP RT blocking queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<>
        struct CurrentThreadWaitFor<uniform_cuda_hip::detail::QueueUniformCudaHipRtBase>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                uniform_cuda_hip::detail::QueueUniformCudaHipRtBase const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for queues on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
