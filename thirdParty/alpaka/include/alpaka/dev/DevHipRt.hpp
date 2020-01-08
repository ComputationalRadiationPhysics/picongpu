/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

#include <alpaka/core/Hip.hpp>

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
        class PltfHipRt;
    }

    namespace queue
    {
        class QueueHipRtBlocking;
        class QueueHipRtNonBlocking;
    }

    namespace dev
    {
        //#############################################################################
        //! The HIP RT device handle.
        class DevHipRt : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevHipRt>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfHipRt>;

        protected:
            //-----------------------------------------------------------------------------
            DevHipRt() = default;
        public:
            //-----------------------------------------------------------------------------
            DevHipRt(DevHipRt const &) = default;
            //-----------------------------------------------------------------------------
            DevHipRt(DevHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevHipRt const &) -> DevHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevHipRt &&) -> DevHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevHipRt const & rhs) const
            -> bool
            {
                return m_iDevice == rhs.m_iDevice;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC ~DevHipRt() = default;

        public:
            int m_iDevice;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device name get trait specialization.
            template<>
            struct GetName<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevHipRt const & dev)
                -> std::string
                {
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(
                        hipGetDeviceProperties(
                            &hipDevProp,
                            dev.m_iDevice));

                    return std::string(hipDevProp.name);
                }
            };

            //#############################################################################
            //! The HIP RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    // \TODO: Check which is faster: hipMemGetInfo().totalInternal vs hipGetDeviceProperties().totalGlobalMem
                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The HIP RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevHipRt const & dev)
                -> std::size_t
                {
                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));

                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_HIP_RT_CHECK(
                        hipMemGetInfo(
                            &freeInternal,
                            &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The HIP RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(
                        hipDeviceReset());
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
            class BufHipRt;

            namespace traits
            {
                //#############################################################################
                //! The HIP RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevHipRt,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufHipRt<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevHipRt>
            {
                using type = pltf::PltfHipRt;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread HIP device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Set the current device to wait for.
                    ALPAKA_HIP_RT_CHECK(hipSetDevice(
                        dev.m_iDevice));
                    ALPAKA_HIP_RT_CHECK(hipDeviceSynchronize());
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
                dev::DevHipRt,
                queue::Blocking
            >
            {
                using type = queue::QueueHipRtBlocking;
            };

            template<>
            struct QueueType<
                dev::DevHipRt,
                queue::NonBlocking
            >
            {
                using type = queue::QueueHipRtNonBlocking;
            };
        }
    }
}

#endif
