/* Copyright 2022 Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <condition_variable>
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <stdexcept>
#    include <thread>

namespace alpaka
{
    class EventUniformCudaHipRt;

    namespace uniform_cuda_hip::detail
    {
        //! The CUDA/HIP RT queue implementation.
        class QueueUniformCudaHipRtImpl final
        {
        public:
            ALPAKA_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt const& dev)
                : m_dev(dev)
                , m_UniformCudaHipQueue()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_dev.getNativeHandle()));

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
            QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl&&) = default;
            auto operator=(QueueUniformCudaHipRtImpl&&) -> QueueUniformCudaHipRtImpl& = delete;
            ALPAKA_FN_HOST ~QueueUniformCudaHipRtImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // In case the device is still doing work in the queue when cuda/hip-StreamDestroy() is called, the
                // function will return immediately and the resources associated with queue will be released
                // automatically once the device has completed all work in queue.
                // -> No need to synchronize here.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ALPAKA_API_PREFIX(StreamDestroy)(m_UniformCudaHipQueue));
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

        public:
            DevUniformCudaHipRt const m_dev; //!< The device this queue is bound to.
        private:
            ALPAKA_API_PREFIX(Stream_t) m_UniformCudaHipQueue;
        };

        //! The CUDA/HIP RT queue.
        template<bool TBlocking>
        class QueueUniformCudaHipRt final
            : public concepts::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRt<TBlocking>>
            , public concepts::Implements<ConceptQueue, QueueUniformCudaHipRt<TBlocking>>
            , public concepts::Implements<ConceptGetDev, QueueUniformCudaHipRt<TBlocking>>
        {
        public:
            ALPAKA_FN_HOST QueueUniformCudaHipRt(DevUniformCudaHipRt const& dev)
                : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl>(dev))
            {
            }
            ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRt const& rhs) const -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRt const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_spQueueImpl->getNativeHandle();
            }

        public:
            std::shared_ptr<QueueUniformCudaHipRtImpl> m_spQueueImpl;
        };

    } // namespace uniform_cuda_hip::detail

    namespace trait
    {
        //! The CUDA/HIP RT queue device type trait specialization.
        template<bool TBlocking>
        struct DevType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            using type = DevUniformCudaHipRt;
        };

        //! The CUDA/HIP RT queue event type trait specialization.
        template<bool TBlocking>
        struct EventType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            using type = EventUniformCudaHipRt;
        };

        //! The CUDA/HIP RT queue device get trait specialization.
        template<bool TBlocking>
        struct GetDev<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            ALPAKA_FN_HOST static auto getDev(uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking> const& queue)
                -> DevUniformCudaHipRt
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT queue test trait specialization.
        template<bool TBlocking>
        struct Empty<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            ALPAKA_FN_HOST static auto empty(uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking> const& queue)
                -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for queues on non current device.
                ALPAKA_API_PREFIX(Error_t) ret = ALPAKA_API_PREFIX(Success);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = ALPAKA_API_PREFIX(StreamQuery)(queue.getNativeHandle()),
                    ALPAKA_API_PREFIX(ErrorNotReady));
                return (ret == ALPAKA_API_PREFIX(Success));
            }
        };

        //! The CUDA/HIP RT queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<bool TBlocking>
        struct CurrentThreadWaitFor<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking> const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for queues on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamSynchronize)(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT queue enqueue trait specialization.
        template<bool TBlocking, typename TTask>
        struct Enqueue<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>, TTask>
        {
            enum class CallbackState
            {
                enqueued,
                notified,
                finished,
            };

            struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
            {
                std::mutex m_mutex;
                std::condition_variable m_event;
                CallbackState m_state = CallbackState::enqueued;
            };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            static void CUDART_CB
#    else
            static void HIPRT_CB
#    endif
            uniformCudaHipRtCallback(
#    if !defined(CUDA_VERSION) || CUDA_VERSION <= 9020
                ALPAKA_API_PREFIX(Stream_t) /*queue*/,
                ALPAKA_API_PREFIX(Error_t) /*status*/,
#    endif
                void* arg)

            {
                // explicitly copy the shared_ptr so that this method holds the state even when the executing thread
                // has already finished.
                const auto spCallbackSynchronizationData
                    = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                // Notify the executing thread.
                {
                    std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
                    spCallbackSynchronizationData->m_state = CallbackState::notified;
                }
                spCallbackSynchronizationData->m_event.notify_one();

                // Wait for the executing thread to finish the task if it has not already finished.
                std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
                if(spCallbackSynchronizationData->m_state != CallbackState::finished)
                {
                    spCallbackSynchronizationData->m_event.wait(
                        lock,
                        [&spCallbackSynchronizationData]()
                        { return spCallbackSynchronizationData->m_state == CallbackState::finished; });
                }
            }

            ALPAKA_FN_HOST static auto enqueue(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>& queue,
                TTask const& task) -> void
            {
                auto spCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && CUDA_VERSION > 9020
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(LaunchHostFunc)(
                    queue.getNativeHandle(),
                    uniformCudaHipRtCallback,
                    spCallbackSynchronizationData.get()));
#    else
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(StreamAddCallback)(
                    queue.getNativeHandle(),
                    uniformCudaHipRtCallback,
                    spCallbackSynchronizationData.get(),
                    0u));
#    endif

                // We start a new std::thread which stores the task to be executed.
                // This circumvents the limitation that it is not possible to call CUDA/HIP methods within the CUDA/HIP
                // callback thread. The CUDA/HIP thread signals the std::thread when it is ready to execute the task.
                // The CUDA/HIP thread is waiting for the std::thread to signal that it is finished executing the task
                // before it executes the next task in the queue (CUDA/HIP stream).
                std::thread t(
                    [spCallbackSynchronizationData, task]()
                    {
                        // If the callback has not yet been called, we wait for it.
                        {
                            std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
                            if(spCallbackSynchronizationData->m_state != CallbackState::notified)
                            {
                                spCallbackSynchronizationData->m_event.wait(
                                    lock,
                                    [&spCallbackSynchronizationData]()
                                    { return spCallbackSynchronizationData->m_state == CallbackState::notified; });
                            }

                            task();

                            // Notify the waiting CUDA/HIP thread.
                            spCallbackSynchronizationData->m_state = CallbackState::finished;
                        }
                        spCallbackSynchronizationData->m_event.notify_one();
                    });

                if constexpr(TBlocking)
                {
                    t.join();
                }
                else
                {
                    t.detach();
                }
            }
        };

        //! The CUDA/HIP RT queue native handle trait specialization.
        template<bool TBlocking>
        struct NativeHandle<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking>>
        {
            [[nodiscard]] static auto getNativeHandle(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TBlocking> const& queue)
            {
                return queue.getNativeHandle();
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
