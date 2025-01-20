/* Copyright 2024 Benjamin Worpitz, Jakob Krude, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 *                Antonio Di Pilato, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dev/common/QueueRegistry.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Properties.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <cstddef>
#include <string>
#include <vector>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait

    namespace uniform_cuda_hip::detail
    {
        template<typename TApi, bool TBlocking>
        class QueueUniformCudaHipRt;
    } // namespace uniform_cuda_hip::detail

    template<typename TApi>
    using QueueUniformCudaHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, true>;

    template<typename TApi>
    using QueueUniformCudaHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, false>;

    template<typename TApi>
    struct PlatformUniformCudaHipRt;

    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct BufUniformCudaHipRt;

    //! The CUDA/HIP RT device handle.
    template<typename TApi>
    class DevUniformCudaHipRt
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevUniformCudaHipRt<TApi>>
        , public concepts::Implements<ConceptDev, DevUniformCudaHipRt<TApi>>
    {
        friend struct trait::GetDevByIdx<PlatformUniformCudaHipRt<TApi>>;

        using IDeviceQueue = uniform_cuda_hip::detail::QueueUniformCudaHipRtImpl<TApi>;

    protected:
        DevUniformCudaHipRt() : m_QueueRegistry{std::make_shared<alpaka::detail::QueueRegistry<IDeviceQueue>>()}
        {
        }

    public:
        ALPAKA_FN_HOST auto operator==(DevUniformCudaHipRt const& rhs) const -> bool
        {
            return m_iDevice == rhs.m_iDevice;
        }

        ALPAKA_FN_HOST auto operator!=(DevUniformCudaHipRt const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept -> int
        {
            return m_iDevice;
        }

        [[nodiscard]] ALPAKA_FN_HOST auto getAllQueues() const -> std::vector<std::shared_ptr<IDeviceQueue>>
        {
            return m_QueueRegistry->getAllExistingQueues();
        }

        //! Registers the given queue on this device.
        //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<IDeviceQueue> spQueue) const -> void
        {
            m_QueueRegistry->registerQueue(spQueue);
        }

    private:
        DevUniformCudaHipRt(int iDevice)
            : m_iDevice(iDevice)
            , m_QueueRegistry(std::make_shared<alpaka::detail::QueueRegistry<IDeviceQueue>>())
        {
        }

        int m_iDevice;

        std::shared_ptr<alpaka::detail::QueueRegistry<IDeviceQueue>> m_QueueRegistry;
    };

    namespace trait
    {
        //! The CUDA/HIP RT device name get trait specialization.
        template<typename TApi>
        struct GetName<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getName(DevUniformCudaHipRt<TApi> const& dev) -> std::string
            {
                // There is cuda/hip-DeviceGetAttribute as faster alternative to cuda/hip-GetDeviceProperties to get a
                // single device property but it has no option to get the name
                typename TApi::DeviceProp_t devProp;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&devProp, dev.getNativeHandle()));

                return std::string(devProp.name);
            }
        };

        //! The CUDA/HIP RT device available memory get trait specialization.
        template<typename TApi>
        struct GetMemBytes<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevUniformCudaHipRt<TApi> const& dev) -> std::size_t
            {
                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

                std::size_t freeInternal(0u);
                std::size_t totalInternal(0u);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memGetInfo(&freeInternal, &totalInternal));

                return totalInternal;
            }
        };

        //! The CUDA/HIP RT device free memory get trait specialization.
        template<typename TApi>
        struct GetFreeMemBytes<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevUniformCudaHipRt<TApi> const& dev) -> std::size_t
            {
                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

                std::size_t freeInternal(0u);
                std::size_t totalInternal(0u);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memGetInfo(&freeInternal, &totalInternal));

                return freeInternal;
            }
        };

        //! The CUDA/HIP RT device warp size get trait specialization.
        template<typename TApi>
        struct GetWarpSizes<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevUniformCudaHipRt<TApi> const& dev) -> std::vector<std::size_t>
            {
                return {GetPreferredWarpSize<DevUniformCudaHipRt<TApi>>::getPreferredWarpSize(dev)};
            }
        };

        //! The CUDA/HIP RT preferred device warp size get trait specialization.
        template<typename TApi>
        struct GetPreferredWarpSize<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getPreferredWarpSize(DevUniformCudaHipRt<TApi> const& dev) -> std::size_t
            {
                int warpSize = 0;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::deviceGetAttribute(&warpSize, TApi::deviceAttributeWarpSize, dev.getNativeHandle()));
                return static_cast<std::size_t>(warpSize);
            }
        };

#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        //! The CUDA RT preferred device warp size get trait specialization.
        template<>
        struct GetPreferredWarpSize<DevUniformCudaHipRt<ApiCudaRt>>
        {
            ALPAKA_FN_HOST static constexpr auto getPreferredWarpSize(DevUniformCudaHipRt<ApiCudaRt> const& /* dev */)
                -> std::size_t
            {
                // All CUDA GPUs to date have a warp size of 32 threads.
                return 32u;
            }
        };
#    endif // ALPAKA_ACC_GPU_CUDA_ENABLED

        //! The CUDA/HIP RT device reset trait specialization.
        template<typename TApi>
        struct Reset<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto reset(DevUniformCudaHipRt<TApi> const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceReset());
            }
        };

        //! The CUDA/HIP RT device native handle trait specialization.
        template<typename TApi>
        struct NativeHandle<DevUniformCudaHipRt<TApi>>
        {
            [[nodiscard]] static auto getNativeHandle(DevUniformCudaHipRt<TApi> const& dev)
            {
                return dev.getNativeHandle();
            }
        };

        //! The CUDA/HIP RT device memory buffer type trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct BufType<DevUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
        {
            using type = BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>;
        };

        //! The CUDA/HIP RT device platform type trait specialization.
        template<typename TApi>
        struct PlatformType<DevUniformCudaHipRt<TApi>>
        {
            using type = PlatformUniformCudaHipRt<TApi>;
        };

        //! The thread CUDA/HIP device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<typename TApi>
        struct CurrentThreadWaitFor<DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(DevUniformCudaHipRt<TApi> const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Set the current device to wait for.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceSynchronize());
            }
        };

        template<typename TApi>
        struct QueueType<DevUniformCudaHipRt<TApi>, Blocking>
        {
            using type = QueueUniformCudaHipRtBlocking<TApi>;
        };

        template<typename TApi>
        struct QueueType<DevUniformCudaHipRt<TApi>, NonBlocking>
        {
            using type = QueueUniformCudaHipRtNonBlocking<TApi>;
        };
    } // namespace trait
} // namespace alpaka

#endif
