/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Cuda.hpp>

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
        class PltfCudaRt;
    }

    namespace dev
    {
        //#############################################################################
        //! The CUDA RT device handle.
        class DevCudaRt
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfCudaRt>;

        protected:
            //-----------------------------------------------------------------------------
            DevCudaRt() = default;
        public:
            //-----------------------------------------------------------------------------
            DevCudaRt(DevCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            DevCudaRt(DevCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCudaRt const &) -> DevCudaRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCudaRt &&) -> DevCudaRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevCudaRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevCudaRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevCudaRt() = default;

        public:
            int m_iDevice;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device name get trait specialization.
            template<>
            struct GetName<
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCudaRt const & dev)
                -> std::string
                {
                    // There is cudaDeviceGetAttribute as faster alternative to cudaGetDeviceProperties to get a single device property but it has no option to get the name
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(
                        cudaGetDeviceProperties(
                            &cudaDevProp,
                            dev.m_iDevice));

                    return std::string(cudaDevProp.name);
                }
            };

            //#############################################################################
            //! The CUDA RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCudaRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The CUDA RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCudaRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCudaRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(
                        cudaDeviceReset());
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
            class BufCudaRt;

            namespace traits
            {
                //#############################################################################
                //! The CUDA RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevCudaRt,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufCudaRt<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevCudaRt>
            {
                using type = pltf::PltfCudaRt;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread CUDA device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevCudaRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceSynchronize());
                }
            };
        }
    }
}

#endif
