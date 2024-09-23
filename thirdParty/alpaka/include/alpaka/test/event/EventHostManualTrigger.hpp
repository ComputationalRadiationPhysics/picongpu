/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Jan Stephan, Jeffrey Kelling, Andrea Bocci,
 *                Bernhard Manfred Gruber, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#include <condition_variable>
#include <mutex>
#include <utility>

namespace alpaka::test
{
    namespace trait
    {
        template<typename TDev>
        struct EventHostManualTriggerType;

        template<typename TDev>
        struct IsEventHostManualTriggerSupported;
    } // namespace trait

    //! The event host manual trigger type trait alias template to remove the ::type.
    template<typename TDev>
    using EventHostManualTrigger = typename trait::EventHostManualTriggerType<TDev>::type;

    template<typename TDev>
    ALPAKA_FN_HOST auto isEventHostManualTriggerSupported(TDev const& dev) -> bool
    {
        return trait::IsEventHostManualTriggerSupported<TDev>::isSupported(dev);
    }

    namespace cpu::detail
    {
        //! Event that can be enqueued into a queue and can be triggered by the Host.
        template<class TDev = DevCpu>
        class EventHostManualTriggerCpuImpl
        {
        public:
            //! Constructor.
            ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(TDev dev) noexcept
                : m_dev(std::move(dev))
                , m_mutex()
                , m_enqueueCount(0u)
                , m_bIsReady(true)
            {
            }

            EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl const& other) = delete;
            auto operator=(EventHostManualTriggerCpuImpl const&) -> EventHostManualTriggerCpuImpl& = delete;

            void trigger()
            {
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_bIsReady = true;
                }
                m_conditionVariable.notify_one();
                // Give alpaka time to update into the new state, process all events and tasks.
                std::this_thread::sleep_for(std::chrono::milliseconds(200u));
            }

        public:
            TDev const m_dev; //!< The device this event is bound to.

            mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.

            mutable std::condition_variable m_conditionVariable; //!< The condition signaling the event completion.
            std::size_t m_enqueueCount; //!< The number of times this event has been enqueued.

            bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                             //!< completed).
        };
    } // namespace cpu::detail

    //! Event that can be enqueued into a queue and can be triggered by the Host.
    template<class TDev = DevCpu>
    class EventHostManualTriggerCpu
    {
    public:
        //! Constructor.
        ALPAKA_FN_HOST EventHostManualTriggerCpu(TDev const& dev)
            : m_spEventImpl(std::make_shared<cpu::detail::EventHostManualTriggerCpuImpl<TDev>>(dev))
        {
        }

        //! Equality comparison operator.
        ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCpu const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }

        //! Inequality comparison operator.
        ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCpu const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        void trigger()
        {
            m_spEventImpl->trigger();
            // Give alpaka time to update into the new state, process all events and tasks.
            std::this_thread::sleep_for(std::chrono::milliseconds(200u));
        }

    public:
        std::shared_ptr<cpu::detail::EventHostManualTriggerCpuImpl<TDev>> m_spEventImpl;
    };

    namespace trait
    {
        template<>
        struct EventHostManualTriggerType<DevCpu>
        {
            using type = test::EventHostManualTriggerCpu<DevCpu>;
        };

        //! The CPU event host manual trigger support get trait specialization.
        template<>
        struct IsEventHostManualTriggerSupported<DevCpu>
        {
            ALPAKA_FN_HOST static auto isSupported(DevCpu const&) -> bool
            {
                return true;
            }
        };
    } // namespace trait
} // namespace alpaka::test

namespace alpaka::trait
{
    //! The CPU device event device get trait specialization.
    template<typename TDev>
    struct GetDev<test::EventHostManualTriggerCpu<TDev>>
    {
        //
        ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCpu<TDev> const& event) -> TDev
        {
            return event.m_spEventImpl->m_dev;
        }
    };

    //! The CPU device event test trait specialization.
    template<typename TDev>
    struct IsComplete<test::EventHostManualTriggerCpu<TDev>>
    {
        //! \return If the event is not waiting within a queue (not enqueued or already handled).
        ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCpu<TDev> const& event) -> bool
        {
            std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

            return event.m_spEventImpl->m_bIsReady;
        }
    };

    template<typename TDev>
    struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
    {
        //
        ALPAKA_FN_HOST static auto enqueue(
            QueueGenericThreadsNonBlocking<TDev>& queue,
            test::EventHostManualTriggerCpu<TDev>& event) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl = event.m_spEventImpl;

            // Setting the event state and enqueuing it has to be atomic.
            std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // Increment the enqueue counter. This is used to skip waits for events that had already been finished
            // and re-enqueued which would lead to deadlocks.
            ++spEventImpl->m_enqueueCount;

            auto const enqueueCount = spEventImpl->m_enqueueCount;

            // Enqueue a task that only resets the events flag if it is completed.
            queue.m_spQueueImpl->m_workerThread.submit(
                [spEventImpl, enqueueCount]() mutable
                {
                    std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
                    spEventImpl->m_conditionVariable.wait(
                        lk2,
                        [spEventImpl, enqueueCount]
                        { return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady; });
                });
        }
    };

    template<typename TDev>
    struct Enqueue<QueueGenericThreadsBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
    {
        //
        ALPAKA_FN_HOST static auto enqueue(
            QueueGenericThreadsBlocking<TDev>&,
            test::EventHostManualTriggerCpu<TDev>& event) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl = event.m_spEventImpl;

            // Setting the event state and enqueuing it has to be atomic.
            std::unique_lock<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // Increment the enqueue counter. This is used to skip waits for events that had already been finished
            // and re-enqueued which would lead to deadlocks.
            ++spEventImpl->m_enqueueCount;

            auto const enqueueCount = spEventImpl->m_enqueueCount;

            spEventImpl->m_conditionVariable.wait(
                lk,
                [spEventImpl, enqueueCount]
                { return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady; });
        }
    };
} // namespace alpaka::trait

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include "alpaka/core/BoostPredef.hpp"

#    include <cuda.h>

#    if !BOOST_LANG_CUDA && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    include "alpaka/core/Cuda.hpp"

namespace alpaka::test
{
    namespace uniform_cuda_hip::detail
    {
        class EventHostManualTriggerCudaImpl final
        {
            using TApi = alpaka::ApiCudaRt;

        public:
            ALPAKA_FN_HOST EventHostManualTriggerCudaImpl(DevCudaRt const& dev)
                : m_dev(dev)
                , m_mutex()
                , m_bIsReady(true)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.getNativeHandle()));
                // Allocate the buffer on this device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    cudaMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
            }

            EventHostManualTriggerCudaImpl(EventHostManualTriggerCudaImpl const&) = delete;
            auto operator=(EventHostManualTriggerCudaImpl const&) -> EventHostManualTriggerCudaImpl& = delete;

            ALPAKA_FN_HOST ~EventHostManualTriggerCudaImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Free the buffer.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cudaFree(m_devMem));
            }

            void trigger()
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_bIsReady = true;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.getNativeHandle()));
                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    cudaMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
                // Give alpaka time to update into the new state, process all events and tasks.
                std::this_thread::sleep_for(std::chrono::milliseconds(200u));
            }

        public:
            DevCudaRt const m_dev; //!< The device this event is bound to.

            mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.
            void* m_devMem;

            bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                             //!< completed).
        };
    } // namespace uniform_cuda_hip::detail

    class EventHostManualTriggerCuda final
    {
    public:
        ALPAKA_FN_HOST EventHostManualTriggerCuda(DevCudaRt const& dev)
            : m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl>(dev))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }

        ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCuda const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }

        ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCuda const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        void trigger()
        {
            m_spEventImpl->trigger();
            // Give alpaka time to update into the new state, process all events and tasks.
            std::this_thread::sleep_for(std::chrono::milliseconds(200u));
        }

    public:
        std::shared_ptr<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl> m_spEventImpl;
    };

    namespace trait
    {
        template<>
        struct EventHostManualTriggerType<DevCudaRt>
        {
            using type = test::EventHostManualTriggerCuda;
        };

        //! The CPU event host manual trigger support get trait specialization.
        template<>
        struct IsEventHostManualTriggerSupported<DevCudaRt>
        {
            ALPAKA_FN_HOST static auto isSupported([[maybe_unused]] DevCudaRt const& dev) -> bool
            {
#    if CUDA_VERSION < 11070
                int result = 0;
                cuDeviceGetAttribute(&result, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev.getNativeHandle());
                return result != 0;
#    else
                return true; // Always enabled as of CUDA 11.7
#    endif
            }
        };
    } // namespace trait
} // namespace alpaka::test

namespace alpaka::trait
{
    namespace detail
    {
        // TODO: Replace with cuStreamWaitValue32 once support for CUDA < 12 is dropped.
        inline auto streamWaitValue(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
            -> CUresult
        {
            // NVIDIA introduced a new stream memory ops API with CUDA 11.7 (called v2). The corresponding CUDA
            // functions were suffixed with `_v2`. With CUDA 12.0 v1 of the API was removed and the `_v2` removed
            // from the new functions. So CUDA <= 11.6 and CUDA >= 12.0 share the same function signature but
            // internally do different things.
#    if(CUDA_VERSION < 11070) || (CUDA_VERSION >= 12000)
            return cuStreamWaitValue32(stream, addr, value, flags);
#    else
            return cuStreamWaitValue32_v2(stream, addr, value, flags);
#    endif
        }
    } // namespace detail

    //! The CPU device event device get trait specialization.
    template<>
    struct GetDev<test::EventHostManualTriggerCuda>
    {
        ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCuda const& event) -> DevCudaRt
        {
            return event.m_spEventImpl->m_dev;
        }
    };

    //! The CPU device event test trait specialization.
    template<>
    struct IsComplete<test::EventHostManualTriggerCuda>
    {
        //! \return If the event is not waiting within a queue (not enqueued or already handled).
        ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCuda const& event) -> bool
        {
            std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

            return event.m_spEventImpl->m_bIsReady;
        }
    };

    template<>
    struct Enqueue<QueueCudaRtNonBlocking, test::EventHostManualTriggerCuda>
    {
        ALPAKA_FN_HOST static auto enqueue(QueueCudaRtNonBlocking& queue, test::EventHostManualTriggerCuda& event)
            -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl(event.m_spEventImpl);

            // Setting the event state and enqueuing it has to be atomic.
            std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // PGI Profiler`s User Guide:
            // The following are known issues related to Events and Metrics:
            // * In event or metric profiling, kernel launches are blocking. Thus kernels waiting
            //   on host updates may hang. This includes synchronization between the host and
            //   the device build upon value-based CUDA queue synchronization APIs such as
            //   cuStreamWaitValue32() and cuStreamWriteValue32().
            ALPAKA_CUDA_DRV_CHECK(detail::streamWaitValue(
                static_cast<CUstream>(queue.getNativeHandle()),
                reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
                0x0101'0101u,
                CU_STREAM_WAIT_VALUE_GEQ));
        }
    };

    template<>
    struct Enqueue<QueueCudaRtBlocking, test::EventHostManualTriggerCuda>
    {
        ALPAKA_FN_HOST static auto enqueue(QueueCudaRtBlocking& queue, test::EventHostManualTriggerCuda& event) -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl(event.m_spEventImpl);

            // Setting the event state and enqueuing it has to be atomic.
            std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // PGI Profiler`s User Guide:
            // The following are known issues related to Events and Metrics:
            // * In event or metric profiling, kernel launches are blocking. Thus kernels waiting
            //   on host updates may hang. This includes synchronization between the host and
            //   the device build upon value-based CUDA queue synchronization APIs such as
            //   cuStreamWaitValue32() and cuStreamWriteValue32().
            ALPAKA_CUDA_DRV_CHECK(detail::streamWaitValue(
                static_cast<CUstream>(queue.getNativeHandle()),
                reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
                0x0101'0101u,
                CU_STREAM_WAIT_VALUE_GEQ));
        }
    };
} // namespace alpaka::trait
#endif


#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <hip/hip_runtime.h>

#    if !BOOST_LANG_HIP && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include "alpaka/core/Hip.hpp"

namespace alpaka::test
{
    namespace hip::detail
    {
        class EventHostManualTriggerHipImpl final
        {
            using TApi = alpaka::ApiHipRt;

        public:
            ALPAKA_FN_HOST EventHostManualTriggerHipImpl(DevHipRt const& dev) : m_dev(dev), m_mutex(), m_bIsReady(true)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.getNativeHandle()));
                // Allocate the buffer on this device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    hipMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
            }

            EventHostManualTriggerHipImpl(EventHostManualTriggerHipImpl const&) = delete;
            auto operator=(EventHostManualTriggerHipImpl const&) -> EventHostManualTriggerHipImpl& = delete;

            ALPAKA_FN_HOST ~EventHostManualTriggerHipImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Free the buffer.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(hipFree(m_devMem));
            }

            void trigger()
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_bIsReady = true;

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.getNativeHandle()));
                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    hipMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
                // Give alpaka time to update into the new state, process all events and tasks.
                std::this_thread::sleep_for(std::chrono::milliseconds(200u));
            }

        public:
            DevHipRt const m_dev; //!< The device this event is bound to.

            mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.
            void* m_devMem;

            bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                             //!< completed).
        };
    } // namespace hip::detail

    class EventHostManualTriggerHip final
    {
    public:
        ALPAKA_FN_HOST EventHostManualTriggerHip(DevHipRt const& dev)
            : m_spEventImpl(std::make_shared<hip::detail::EventHostManualTriggerHipImpl>(dev))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }

        ALPAKA_FN_HOST auto operator==(EventHostManualTriggerHip const& rhs) const -> bool
        {
            return (m_spEventImpl == rhs.m_spEventImpl);
        }

        ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerHip const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        void trigger()
        {
            m_spEventImpl->trigger();
            // Give alpaka time to update into the new state, process all events and tasks.
            std::this_thread::sleep_for(std::chrono::milliseconds(200u));
        }

    public:
        std::shared_ptr<hip::detail::EventHostManualTriggerHipImpl> m_spEventImpl;
    };

    namespace trait
    {
        template<>
        struct EventHostManualTriggerType<DevHipRt>
        {
            using type = test::EventHostManualTriggerHip;
        };

        //! The HIP event host manual trigger support get trait specialization.
        template<>
        struct IsEventHostManualTriggerSupported<DevHipRt>
        {
            // TODO: there is no CUDA_VERSION in the HIP compiler path.
            // TODO: there is a hipDeviceGetAttribute, but there is no pendant for
            // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.
            ALPAKA_FN_HOST static auto isSupported(DevHipRt const&) -> bool
            {
                return false;
            }
        };
    } // namespace trait
} // namespace alpaka::test

namespace alpaka::trait
{
    //! The CPU device event device get trait specialization.
    template<>
    struct GetDev<test::EventHostManualTriggerHip>
    {
        ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerHip const& event) -> DevHipRt
        {
            return event.m_spEventImpl->m_dev;
        }
    };

    //! The CPU device event test trait specialization.
    template<>
    struct IsComplete<test::EventHostManualTriggerHip>
    {
        //! \return If the event is not waiting within a queue (not enqueued or already handled).
        ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerHip const& event) -> bool
        {
            std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

            return event.m_spEventImpl->m_bIsReady;
        }
    };

    template<>
    struct Enqueue<QueueHipRtNonBlocking, test::EventHostManualTriggerHip>
    {
        using TApi = alpaka::ApiHipRt;

        ALPAKA_FN_HOST static auto enqueue(QueueHipRtNonBlocking& queue, test::EventHostManualTriggerHip& event)
            -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl(event.m_spEventImpl);

            // Setting the event state and enqueuing it has to be atomic.
            std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // PGI Profiler`s User Guide:
            // The following are known issues related to Events and Metrics:
            // * In event or metric profiling, kernel launches are blocking. Thus kernels waiting
            //   on host updates may hang. This includes synchronization between the host and
            //   the device build upon value-based CUDA queue synchronization APIs such as
            //   cuStreamWaitValue32() and cuStreamWriteValue32().
            int32_t hostMem = 0;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cerr << "[Workaround] polling of device-located value in stream, as hipStreamWaitValue32 is not "
                         "available.\n";
#    endif
            while(hostMem < 0x0101'0101)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMemcpyDtoHAsync(
                    &hostMem,
                    reinterpret_cast<hipDeviceptr_t>(event.m_spEventImpl->m_devMem),
                    sizeof(int32_t),
                    queue.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipStreamSynchronize(queue.getNativeHandle()));
            }
        }
    };

    template<>
    struct Enqueue<QueueHipRtBlocking, test::EventHostManualTriggerHip>
    {
        using TApi = alpaka::ApiHipRt;

        ALPAKA_FN_HOST static auto enqueue(QueueHipRtBlocking& /* queue */, test::EventHostManualTriggerHip& event)
            -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
            auto spEventImpl(event.m_spEventImpl);

            // Setting the event state and enqueuing it has to be atomic.
            std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

            // The event should not yet be enqueued.
            ALPAKA_ASSERT(spEventImpl->m_bIsReady);

            // Set its state to enqueued.
            spEventImpl->m_bIsReady = false;

            // PGI Profiler`s User Guide:
            // The following are known issues related to Events and Metrics:
            // * In event or metric profiling, kernel launches are blocking. Thus kernels waiting
            //   on host updates may hang. This includes synchronization between the host and
            //   the device build upon value-based HIP queue synchronization APIs such as
            //   cuStreamWaitValue32() and cuStreamWriteValue32().

            // workaround for missing cuStreamWaitValue32 in HIP
            std::uint32_t hmem = 0;
            do
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10u));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    hipMemcpy(&hmem, event.m_spEventImpl->m_devMem, sizeof(std::uint32_t), hipMemcpyDefault));
            } while(hmem < 0x0101'0101u);
        }
    };
} // namespace alpaka::trait
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
namespace alpaka
{
    namespace test
    {
        template<typename TTag>
        class EventHostManualTriggerSycl
        {
        public:
            EventHostManualTriggerSycl(DevGenericSycl<TTag> const&)
            {
            }

            auto trigger()
            {
            }
        };

        namespace trait
        {
            template<typename TTag>
            struct EventHostManualTriggerType<DevGenericSycl<TTag>>
            {
                using type = alpaka::test::EventHostManualTriggerSycl<TTag>;
            };

            template<typename TTag>
            struct IsEventHostManualTriggerSupported<DevGenericSycl<TTag>>
            {
                ALPAKA_FN_HOST static auto isSupported(DevGenericSycl<TTag> const&) -> bool
                {
                    return false;
                }
            };
        } // namespace trait
    } // namespace test

    namespace trait
    {
        template<typename TTag>
        struct Enqueue<QueueGenericSyclBlocking<TTag>, test::EventHostManualTriggerSycl<TTag>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueGenericSyclBlocking<TTag>& /* queue */,
                test::EventHostManualTriggerSycl<TTag>& /* event */) -> void
            {
            }
        };

        template<typename TTag>
        struct Enqueue<QueueGenericSyclNonBlocking<TTag>, test::EventHostManualTriggerSycl<TTag>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueGenericSyclNonBlocking<TTag>& /* queue */,
                test::EventHostManualTriggerSycl<TTag>& /* event */) -> void
            {
            }
        };

        template<typename TTag>
        struct IsComplete<test::EventHostManualTriggerSycl<TTag>>
        {
            ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerSycl<TTag> const& /* event */) -> bool
            {
                return true;
            }
        };
    } // namespace trait
} // namespace alpaka
#endif
