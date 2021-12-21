/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <condition_variable>
#include <mutex>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    //-----------------------------------------------------------------------------
    namespace test
    {
        namespace traits
        {
            //#############################################################################
            //!
            //#############################################################################
            template<typename TDev>
            struct EventHostManualTriggerType;
            //#############################################################################
            //!
            //#############################################################################
            template<typename TDev>
            struct IsEventHostManualTriggerSupported;
        } // namespace traits

        //#############################################################################
        //! The event host manual trigger type trait alias template to remove the ::type.
        //#############################################################################
        template<typename TDev>
        using EventHostManualTrigger = typename traits::EventHostManualTriggerType<TDev>::type;

        //-----------------------------------------------------------------------------
        template<typename TDev>
        ALPAKA_FN_HOST auto isEventHostManualTriggerSupported(TDev const& dev) -> bool
        {
            return traits::IsEventHostManualTriggerSupported<TDev>::isSupported(dev);
        }

        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! Event that can be enqueued into a queue and can be triggered by the Host.
                //#############################################################################
                template<class TDev = DevCpu>
                class EventHostManualTriggerCpuImpl
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(TDev const& dev) noexcept
                        : m_dev(dev)
                        , m_mutex()
                        , m_enqueueCount(0u)
                        , m_bIsReady(true)
                    {
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl const& other) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl&&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerCpuImpl const&) -> EventHostManualTriggerCpuImpl& = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerCpuImpl&&) -> EventHostManualTriggerCpuImpl& = delete;

                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    void trigger()
                    {
                        {
                            std::unique_lock<std::mutex> lock(m_mutex);
                            m_bIsReady = true;
                        }
                        m_conditionVariable.notify_one();
                    }

                public:
                    TDev const m_dev; //!< The device this event is bound to.

                    mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.

                    mutable std::condition_variable
                        m_conditionVariable; //!< The condition signaling the event completion.
                    std::size_t m_enqueueCount; //!< The number of times this event has been enqueued.

                    bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                                     //!< completed).
                };
            } // namespace detail
        } // namespace cpu

        //#############################################################################
        //! Event that can be enqueued into a queue and can be triggered by the Host.
        //#############################################################################
        template<class TDev = DevCpu>
        class EventHostManualTriggerCpu
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHostManualTriggerCpu(TDev const& dev)
                : m_spEventImpl(std::make_shared<cpu::detail::EventHostManualTriggerCpuImpl<TDev>>(dev))
            {
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            EventHostManualTriggerCpu(EventHostManualTriggerCpu const&) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            EventHostManualTriggerCpu(EventHostManualTriggerCpu&&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerCpu const&) -> EventHostManualTriggerCpu& = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerCpu&&) -> EventHostManualTriggerCpu& = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCpu const& rhs) const -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCpu const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }

            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            void trigger()
            {
                m_spEventImpl->trigger();
            }

        public:
            std::shared_ptr<cpu::detail::EventHostManualTriggerCpuImpl<TDev>> m_spEventImpl;
        };

        namespace traits
        {
            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct EventHostManualTriggerType<alpaka::DevCpu>
            {
                using type = alpaka::test::EventHostManualTriggerCpu<DevCpu>;
            };
#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct EventHostManualTriggerType<alpaka::DevOmp5>
            {
                using type = alpaka::test::EventHostManualTriggerCpu<alpaka::DevOmp5>;
            };
#elif defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)
            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct EventHostManualTriggerType<alpaka::DevOacc>
            {
                using type = alpaka::test::EventHostManualTriggerCpu<alpaka::DevOacc>;
            };
#endif
            //#############################################################################
            //! The CPU event host manual trigger support get trait specialization.
            template<>
            struct IsEventHostManualTriggerSupported<alpaka::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto isSupported(alpaka::DevCpu const&) -> bool
                {
                    return true;
                }
            };
#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            //#############################################################################
            //! The Omp5 event host manual trigger support get trait specialization.
            template<>
            struct IsEventHostManualTriggerSupported<alpaka::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto isSupported(alpaka::DevOmp5 const&) -> bool
                {
                    return true;
                }
            };
#elif defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)
            //#############################################################################
            //! The OpenACC event host manual trigger support get trait specialization.
            template<>
            struct IsEventHostManualTriggerSupported<alpaka::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto isSupported(alpaka::DevOacc const&) -> bool
                {
                    return true;
                }
            };
#endif
        } // namespace traits
    } // namespace test
    namespace traits
    {
        //#############################################################################
        //! The CPU device event device get trait specialization.
        //#############################################################################
        template<typename TDev>
        struct GetDev<test::EventHostManualTriggerCpu<TDev>>
        {
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCpu<TDev> const& event) -> TDev
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CPU device event test trait specialization.
        //#############################################################################
        template<typename TDev>
        struct IsComplete<test::EventHostManualTriggerCpu<TDev>>
        {
            //-----------------------------------------------------------------------------
            //! \return If the event is not waiting within a queue (not enqueued or already handled).
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCpu<TDev> const& event) -> bool
            {
                std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                return event.m_spEventImpl->m_bIsReady;
            }
        };

        //#############################################################################
        //!
        //#############################################################################
        template<typename TDev>
        struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
        {
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                QueueGenericThreadsNonBlocking<TDev>& queue,
#else
                QueueGenericThreadsNonBlocking<TDev>&,
#endif
                test::EventHostManualTriggerCpu<TDev>& event) -> void
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

                // Increment the enqueue counter. This is used to skip waits for events that had already been finished
                // and re-enqueued which would lead to deadlocks.
                ++spEventImpl->m_enqueueCount;

                // Workaround: Clang can not support this when natively compiling device code. See
                // ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                auto const enqueueCount = spEventImpl->m_enqueueCount;

                // Enqueue a task that only resets the events flag if it is completed.
                queue.m_spQueueImpl->m_workerThread.enqueueTask([spEventImpl, enqueueCount]() {
                    std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
                    spEventImpl->m_conditionVariable.wait(lk2, [spEventImpl, enqueueCount] {
                        return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady;
                    });
                });
#endif
            }
        };
        //#############################################################################
        //!
        //#############################################################################
        template<typename TDev>
        struct Enqueue<QueueGenericThreadsBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
        {
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueGenericThreadsBlocking<TDev>&,
                test::EventHostManualTriggerCpu<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Copy the shared pointer to ensure that the event implementation is alive as long as it is enqueued.
                auto spEventImpl(event.m_spEventImpl);

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

                spEventImpl->m_conditionVariable.wait(lk, [spEventImpl, enqueueCount] {
                    return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady;
                });
            }
        };
    } // namespace traits
} // namespace alpaka

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    include <cuda.h>

#    if !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    include <alpaka/core/Cuda.hpp>

namespace alpaka
{
    namespace test
    {
        namespace uniform_cuda_hip
        {
            namespace detail
            {
                //#############################################################################
                class EventHostManualTriggerCudaImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHostManualTriggerCudaImpl(DevUniformCudaHipRt const& dev)
                        : m_dev(dev)
                        , m_mutex()
                        , m_bIsReady(true)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.m_iDevice));
                        // Allocate the buffer on this device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
                        // Initiate the memory set.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            cudaMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
                    }
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerCudaImpl(EventHostManualTriggerCudaImpl const&) = delete;
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerCudaImpl(EventHostManualTriggerCudaImpl&&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerCudaImpl const&) -> EventHostManualTriggerCudaImpl& = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerCudaImpl&&) -> EventHostManualTriggerCudaImpl& = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventHostManualTriggerCudaImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.m_iDevice));
                        // Free the buffer.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaFree(m_devMem));
                    }

                    //-----------------------------------------------------------------------------
                    void trigger()
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_bIsReady = true;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            cudaMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
                    }

                public:
                    DevUniformCudaHipRt const m_dev; //!< The device this event is bound to.

                    mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.
                    void* m_devMem;

                    bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                                     //!< completed).
                };
            } // namespace detail
        } // namespace uniform_cuda_hip

        //#############################################################################
        class EventHostManualTriggerCuda final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHostManualTriggerCuda(DevUniformCudaHipRt const& dev)
                : m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl>(dev))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventHostManualTriggerCuda(EventHostManualTriggerCuda const&) = default;
            //-----------------------------------------------------------------------------
            EventHostManualTriggerCuda(EventHostManualTriggerCuda&&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerCuda const&) -> EventHostManualTriggerCuda& = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerCuda&&) -> EventHostManualTriggerCuda& = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCuda const& rhs) const -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCuda const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventHostManualTriggerCuda() = default;

            //-----------------------------------------------------------------------------
            void trigger()
            {
                m_spEventImpl->trigger();
            }

        public:
            std::shared_ptr<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl> m_spEventImpl;
        };

        namespace traits
        {
            //#############################################################################
            template<>
            struct EventHostManualTriggerType<alpaka::DevUniformCudaHipRt>
            {
                using type = alpaka::test::EventHostManualTriggerCuda;
            };
            //#############################################################################
            //! The CPU event host manual trigger support get trait specialization.
            template<>
            struct IsEventHostManualTriggerSupported<alpaka::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto isSupported(alpaka::DevCudaRt const& dev) -> bool
                {
                    int result = 0;
                    cuDeviceGetAttribute(&result, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev.m_iDevice);
                    return result != 0;
                }
            };
        } // namespace traits
    } // namespace test
    namespace traits
    {
        //#############################################################################
        //! The CPU device event device get trait specialization.
        template<>
        struct GetDev<test::EventHostManualTriggerCuda>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCuda const& event) -> DevUniformCudaHipRt
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CPU device event test trait specialization.
        template<>
        struct IsComplete<test::EventHostManualTriggerCuda>
        {
            //-----------------------------------------------------------------------------
            //! \return If the event is not waiting within a queue (not enqueued or already handled).
            ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCuda const& event) -> bool
            {
                std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                return event.m_spEventImpl->m_bIsReady;
            }
        };

        //#############################################################################
        template<>
        struct Enqueue<QueueUniformCudaHipRtNonBlocking, test::EventHostManualTriggerCuda>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking& queue,
                test::EventHostManualTriggerCuda& event) -> void
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
                ALPAKA_CUDA_DRV_CHECK(cuStreamWaitValue32(
                    static_cast<CUstream>(queue.m_spQueueImpl->m_UniformCudaHipQueue),
                    reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
                    0x01010101u,
                    CU_STREAM_WAIT_VALUE_GEQ));
            }
        };
        //#############################################################################
        template<>
        struct Enqueue<QueueUniformCudaHipRtBlocking, test::EventHostManualTriggerCuda>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking& queue,
                test::EventHostManualTriggerCuda& event) -> void
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
                ALPAKA_CUDA_DRV_CHECK(cuStreamWaitValue32(
                    static_cast<CUstream>(queue.m_spQueueImpl->m_UniformCudaHipQueue),
                    reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
                    0x01010101u,
                    CU_STREAM_WAIT_VALUE_GEQ));
            }
        };
    } // namespace traits
} // namespace alpaka
#endif


#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <hip/hip_runtime.h>

#    if !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/core/Hip.hpp>

namespace alpaka
{
    namespace test
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                class EventHostManualTriggerHipImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventHostManualTriggerHipImpl(DevHipRt const& dev)
                        : m_dev(dev)
                        , m_mutex()
                        , m_bIsReady(true)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.m_iDevice));
                        // Allocate the buffer on this device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
                        // Initiate the memory set.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            hipMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
                    }
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerHipImpl(EventHostManualTriggerHipImpl const&) = delete;
                    //-----------------------------------------------------------------------------
                    EventHostManualTriggerHipImpl(EventHostManualTriggerHipImpl&&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerHipImpl const&) -> EventHostManualTriggerHipImpl& = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventHostManualTriggerHipImpl&&) -> EventHostManualTriggerHipImpl& = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventHostManualTriggerHipImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.m_iDevice));
                        // Free the buffer.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipFree(m_devMem));
                    }

                    //-----------------------------------------------------------------------------
                    void trigger()
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_bIsReady = true;

                        // Set the current device.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.m_iDevice));
                        // Initiate the memory set.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                            hipMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
                    }

                public:
                    DevHipRt const m_dev; //!< The device this event is bound to.

                    mutable std::mutex m_mutex; //!< The mutex used to synchronize access to the event.
                    void* m_devMem;

                    bool m_bIsReady; //!< If the event is not waiting within a queue (not enqueued or already
                                     //!< completed).
                };
            } // namespace detail
        } // namespace hip

        //#############################################################################
        class EventHostManualTriggerHip final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventHostManualTriggerHip(DevHipRt const& dev)
                : m_spEventImpl(std::make_shared<hip::detail::EventHostManualTriggerHipImpl>(dev))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventHostManualTriggerHip(EventHostManualTriggerHip const&) = default;
            //-----------------------------------------------------------------------------
            EventHostManualTriggerHip(EventHostManualTriggerHip&&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerHip const&) -> EventHostManualTriggerHip& = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventHostManualTriggerHip&&) -> EventHostManualTriggerHip& = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventHostManualTriggerHip const& rhs) const -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerHip const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~EventHostManualTriggerHip() = default;

            //-----------------------------------------------------------------------------
            void trigger()
            {
                m_spEventImpl->trigger();
            }

        public:
            std::shared_ptr<hip::detail::EventHostManualTriggerHipImpl> m_spEventImpl;
        };

        namespace traits
        {
            //#############################################################################
            template<>
            struct EventHostManualTriggerType<alpaka::DevHipRt>
            {
                using type = alpaka::test::EventHostManualTriggerHip;
            };

            //#############################################################################
            //! The HIP event host manual trigger support get trait specialization.
            template<>
            struct IsEventHostManualTriggerSupported<alpaka::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                // TODO: there is no CUDA_VERSION in the HIP compiler path.
                // TODO: there is a hipDeviceGetAttribute, but there is no pendant for
                // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.
                ALPAKA_FN_HOST static auto isSupported(alpaka::DevHipRt const&) -> bool
                {
                    return false;
                }
            };
        } // namespace traits
    } // namespace test
    namespace traits
    {
        //#############################################################################
        //! The CPU device event device get trait specialization.
        template<>
        struct GetDev<test::EventHostManualTriggerHip>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerHip const& event) -> DevHipRt
            {
                return event.m_spEventImpl->m_dev;
            }
        };

        //#############################################################################
        //! The CPU device event test trait specialization.
        template<>
        struct IsComplete<test::EventHostManualTriggerHip>
        {
            //-----------------------------------------------------------------------------
            //! \return If the event is not waiting within a queue (not enqueued or already handled).
            ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerHip const& event) -> bool
            {
                std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

                return event.m_spEventImpl->m_bIsReady;
            }
        };

        //#############################################################################
        template<>
        struct Enqueue<QueueHipRtNonBlocking, test::EventHostManualTriggerHip>
        {
            //-----------------------------------------------------------------------------
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
                while(hostMem < 0x01010101u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMemcpyDtoHAsync(
                        &hostMem,
                        reinterpret_cast<hipDeviceptr_t>(event.m_spEventImpl->m_devMem),
                        sizeof(int32_t),
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipStreamSynchronize(queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            }
        };
        //#############################################################################
        template<>
        struct Enqueue<QueueHipRtBlocking, test::EventHostManualTriggerHip>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueHipRtBlocking& queue, test::EventHostManualTriggerHip& event)
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
#    if BOOST_COMP_NVCC
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipCUResultTohipError(cuStreamWaitValue32(
                    static_cast<CUstream>(queue.m_spQueueImpl->m_UniformCudaHipQueue),
                    reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
                    0x01010101u,
                    CU_STREAM_WAIT_VALUE_GEQ)));
#    else
                // workaround for missing cuStreamWaitValue32 in HIP
                std::uint32_t hmem = 0;
                do
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10u));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        hipMemcpy(&hmem, event.m_spEventImpl->m_devMem, sizeof(std::uint32_t), hipMemcpyDefault));
                } while(hmem < 0x01010101u);

#    endif
            }
        };
    } // namespace traits
} // namespace alpaka
#endif
