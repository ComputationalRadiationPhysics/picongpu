/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

namespace alpaka
{
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfUniformCudaHipRt;
    }
    namespace queue
    {
        class QueueUniformCudaHipRtBlocking;
        class QueueUniformCudaHipRtNonBlocking;
    }

    namespace dev
    {
        //#############################################################################
        //! The CUDA/HIP RT device handle.
        class DevUniformCudaHipRt :
            public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevUniformCudaHipRt>,
            public concepts::Implements<ConceptDev, DevUniformCudaHipRt>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfUniformCudaHipRt>;

        protected:
            //-----------------------------------------------------------------------------
            DevUniformCudaHipRt() = default;
        public:
            //-----------------------------------------------------------------------------
            DevUniformCudaHipRt(DevUniformCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            DevUniformCudaHipRt(DevUniformCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevUniformCudaHipRt const &) -> DevUniformCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevUniformCudaHipRt &&) -> DevUniformCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevUniformCudaHipRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevUniformCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevUniformCudaHipRt() = default;

        public:
            int m_iDevice;
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        using DevCudaRt = DevUniformCudaHipRt;
#else
        using DevHipRt = DevUniformCudaHipRt;
#endif
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT device name get trait specialization.
            template<>
            struct GetName<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevUniformCudaHipRt const & dev)
                -> std::string
                {
                    // There is cuda/hip-DeviceGetAttribute as faster alternative to cuda/hip-GetDeviceProperties to get a single device property but it has no option to get the name
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    cudaDeviceProp devProp;
#else
                    hipDeviceProp_t devProp;
#endif
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(GetDeviceProperties)(
                            &devProp,
                            dev.m_iDevice));

                    return std::string(devProp.name);
                }
            };

            //#############################################################################
            //! The CUDA/HIP RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevUniformCudaHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(SetDevice)(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(MemGetInfo)(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The CUDA/HIP RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevUniformCudaHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(SetDevice)(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(MemGetInfo)(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA/HIP RT device warp size get trait specialization.
            template<>
            struct GetWarpSize<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getWarpSize(
                    dev::DevUniformCudaHipRt const & dev)
                -> std::size_t
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    cudaDeviceProp devProp;
#else
                    hipDeviceProp_t devProp;
#endif
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(GetDeviceProperties)(
                            &devProp,
                            dev.m_iDevice));

                    return static_cast<std::size_t>(devProp.warpSize);
                }
            };

            //#############################################################################
            //! The CUDA/HIP RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevUniformCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(SetDevice)(
                            dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(DeviceReset)());
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufUniformCudaHipRt;

            namespace traits
            {
                //#############################################################################
                //! The CUDA/HIP RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevUniformCudaHipRt,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufUniformCudaHipRt<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA/HIP RT device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevUniformCudaHipRt>
            {
                using type = pltf::PltfUniformCudaHipRt;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread CUDA/HIP device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevUniformCudaHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(
                        dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceSynchronize)());
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevUniformCudaHipRt,
                queue::Blocking
            >
            {
                using type = queue::QueueUniformCudaHipRtBlocking;
            };

            template<>
            struct QueueType<
                dev::DevUniformCudaHipRt,
                queue::NonBlocking
            >
            {
                using type = queue::QueueUniformCudaHipRtNonBlocking;
            };
        }
    }
}

#endif
