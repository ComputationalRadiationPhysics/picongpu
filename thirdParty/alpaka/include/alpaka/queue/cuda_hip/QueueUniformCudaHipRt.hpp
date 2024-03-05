/* Copyright 2022 Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    template<typename TApi>
    class EventUniformCudaHipRt;

    template<typename TApi>
    class DevUniformCudaHipRt;

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

                // Make sure all pending async work is finished before destroying the stream to guarantee determinism.
                // This would not be necessary for plain CUDA/HIP operations, but we can have host functions in the
                // stream, which reference this queue instance and its CallbackThread. Make sure they are done.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::streamSynchronize(m_UniformCudaHipQueue));
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
                dev.registerQueue(m_spQueueImpl);
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
            using QueueImpl = uniform_cuda_hip::detail::QueueUniformCudaHipRtImpl<TApi>;

            struct HostFuncData
            {
                // We don't need to keep the queue alive, because in it's dtor it will synchronize with the CUDA/HIP
                // stream and wait until all host functions and the CallbackThread are done. It's actually an error to
                // copy the queue into the host function. Destroying it here would call CUDA/HIP APIs from the host
                // function. Passing it further to the Callback thread, would make the Callback thread hold a task
                // containing the queue with the CallbackThread itself. Destroying the task if no other queue instance
                // exists will make the CallbackThread join itself and crash.
                QueueImpl& q;
                TTask t;
            };

            ALPAKA_FN_HOST static void uniformCudaHipRtHostFunc(void* arg)
            {
                auto data = std::unique_ptr<HostFuncData>(reinterpret_cast<HostFuncData*>(arg));
                auto& queue = data->q;
                auto f = queue.m_callbackThread.submit([d = std::move(data)] { d->t(); });
                f.wait();
            }

            ALPAKA_FN_HOST static auto enqueue(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
                TTask const& task) -> void
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::launchHostFunc(
                    queue.getNativeHandle(),
                    uniformCudaHipRtHostFunc,
                    new HostFuncData{*queue.m_spQueueImpl, task}));
                if constexpr(TBlocking)
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
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
