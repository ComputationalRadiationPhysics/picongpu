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

#    include <alpaka/core/CallbackThread.hpp>

#    include <condition_variable>
#    include <functional>
#    include <future>
#    include <memory>
#    include <mutex>
#    include <thread>

namespace alpaka
{
    template<typename TApi>
    class EventUniformCudaHipRt;

    namespace uniform_cuda_hip::detail
    {
        //! The CUDA/HIP RT queue implementation.
        template<typename TApi>
        class QueueUniformCudaHipRtImpl final
        {
        public:
            ALPAKA_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt<TApi> const& dev)
                : m_dev(dev)
                , m_UniformCudaHipQueue()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_dev.getNativeHandle()));

                // - [cuda/hip]StreamDefault: Default queue creation flag.
                // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run
                // concurrently with work in queue 0 (the NULL queue),
                //   and that the created queue should perform no implicit synchronization with queue 0.
                // Create the queue on the current device.
                // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka
                // CPU queue. It would be too much work to implement implicit default queue synchronization on CPU.

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamCreateWithFlags(&m_UniformCudaHipQueue, TApi::streamNonBlocking));
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
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::streamDestroy(m_UniformCudaHipQueue));
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

        public:
            DevUniformCudaHipRt<TApi> const m_dev; //!< The device this queue is bound to.
            core::CallbackThread m_callbackThread;

        private:
            typename TApi::Stream_t m_UniformCudaHipQueue;
        };

        //! The CUDA/HIP RT queue.
        template<typename TApi, bool TBlocking>
        class QueueUniformCudaHipRt
            : public concepts::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRt<TApi, TBlocking>>
            , public concepts::Implements<ConceptQueue, QueueUniformCudaHipRt<TApi, TBlocking>>
            , public concepts::Implements<ConceptGetDev, QueueUniformCudaHipRt<TApi, TBlocking>>
        {
        public:
            ALPAKA_FN_HOST QueueUniformCudaHipRt(DevUniformCudaHipRt<TApi> const& dev)
                : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl<TApi>>(dev))
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
            auto getCallbackThread() -> core::CallbackThread&
            {
                return m_spQueueImpl->m_callbackThread;
            }

        public:
            std::shared_ptr<QueueUniformCudaHipRtImpl<TApi>> m_spQueueImpl;
        };
    } // namespace uniform_cuda_hip::detail

    namespace trait
    {
        //! The CUDA/HIP RT queue device get trait specialization.
        template<typename TApi, bool TBlocking>
        struct GetDev<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            ALPAKA_FN_HOST static auto getDev(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
                -> DevUniformCudaHipRt<TApi>
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT queue test trait specialization.
        template<typename TApi, bool TBlocking>
        struct Empty<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            ALPAKA_FN_HOST static auto empty(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for queues on non current device.
                typename TApi::Error_t ret = TApi::success;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = TApi::streamQuery(queue.getNativeHandle()),
                    TApi::errorNotReady);
                return (ret == TApi::success);
            }
        };

        //! The CUDA/HIP RT queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<typename TApi, bool TBlocking>
        struct CurrentThreadWaitFor<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for queues on non current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT blocking queue device type trait specialization.
        template<typename TApi, bool TBlocking>
        struct DevType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT blocking queue event type trait specialization.
        template<typename TApi, bool TBlocking>
        struct EventType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            using type = EventUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT blocking queue enqueue trait specialization.
        template<typename TApi, bool TBlocking, typename TTask>
        struct Enqueue<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>, TTask>
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

            ALPAKA_FN_HOST static void uniformCudaHipRtHostFunc(void* arg)
            {
                // explicitly copy the shared_ptr so that this method holds the state even when the executing thread
                // has already finished.
                auto const spCallbackSynchronizationData
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
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
                TTask const& task) -> void
            {
                auto spCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::launchHostFunc(
                    queue.getNativeHandle(),
                    uniformCudaHipRtHostFunc,
                    spCallbackSynchronizationData.get()));

                // We start a new std::thread which stores the task to be executed.
                // This circumvents the limitation that it is not possible to call CUDA/HIP methods within the CUDA/HIP
                // callback thread. The CUDA/HIP thread signals the std::thread when it is ready to execute the task.
                // The CUDA/HIP thread is waiting for the std::thread to signal that it is finished executing the task
                // before it executes the next task in the queue (CUDA/HIP stream).
                auto f = queue.getCallbackThread().submit(std::packaged_task<void()>(
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
                    }));

                if constexpr(TBlocking)
                {
                    f.wait();
                }
            }
        };

        //! The CUDA/HIP RT blocking queue native handle trait specialization.
        template<typename TApi, bool TBlocking>
        struct NativeHandle<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            [[nodiscard]] static auto getNativeHandle(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
            {
                return queue.getNativeHandle();
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
