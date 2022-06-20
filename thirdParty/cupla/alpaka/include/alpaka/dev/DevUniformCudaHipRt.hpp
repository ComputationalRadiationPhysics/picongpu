/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato, Jan Stephan
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
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Properties.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/traits/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <cstddef>
#    include <string>
#    include <vector>

namespace alpaka
{
    namespace trait
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }

    namespace uniform_cuda_hip::detail
    {
        template<typename TApi, bool TBlocking>
        class QueueUniformCudaHipRt;
    }

    template<typename TApi>
    using QueueUniformCudaHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, true>;

    template<typename TApi>
    using QueueUniformCudaHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, false>;

    template<typename TApi>
    class PltfUniformCudaHipRt;

    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt;

    //! The CUDA/HIP RT device handle.
    template<typename TApi>
    class DevUniformCudaHipRt
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevUniformCudaHipRt<TApi>>
        , public concepts::Implements<ConceptDev, DevUniformCudaHipRt<TApi>>
    {
        friend struct trait::GetDevByIdx<PltfUniformCudaHipRt<TApi>>;

    protected:
        DevUniformCudaHipRt() = default;

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

    private:
        DevUniformCudaHipRt(int iDevice) : m_iDevice(iDevice)
        {
        }
        int m_iDevice;
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
                typename TApi::DeviceProp_t devProp;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&devProp, dev.getNativeHandle()));

                return {static_cast<std::size_t>(devProp.warpSize)};
            }
        };

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
        struct PltfType<DevUniformCudaHipRt<TApi>>
        {
            using type = PltfUniformCudaHipRt<TApi>;
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
