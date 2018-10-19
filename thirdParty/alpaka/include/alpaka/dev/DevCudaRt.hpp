/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

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

                    // \TODO: Check which is faster: cudaMemGetInfo().totalInternal vs cudaGetDeviceProperties().totalGlobalMem
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
                typename TSize>
            class BufCudaRt;

            namespace traits
            {
                //#############################################################################
                //! The CUDA RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct BufType<
                    dev::DevCudaRt,
                    TElem,
                    TDim,
                    TSize>
                {
                    using type = mem::buf::BufCudaRt<TElem, TDim, TSize>;
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
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
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
