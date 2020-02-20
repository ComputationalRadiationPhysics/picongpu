/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/cpu/ICpuQueue.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/cpu/SysInfo.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>

namespace alpaka
{
    namespace queue
    {
        class QueueCpuNonBlocking;
        class QueueCpuBlocking;

        namespace cpu
        {
            namespace detail
            {
                class QueueCpuNonBlockingImpl;
                class QueueCpuBlockingImpl;
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfCpu;
    }
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device implementation.
                class DevCpuImpl
                {
                private:

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto GetAllQueueImpls(
                        std::vector<std::weak_ptr<queue::cpu::ICpuQueue>> & queues) const
                    -> std::vector<std::shared_ptr<queue::cpu::ICpuQueue>>
                    {
                        std::vector<std::shared_ptr<queue::cpu::ICpuQueue>> vspQueues;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto it = queues.begin(); it != queues.end();)
                        {
                            auto spQueue(it->lock());
                            if(spQueue)
                            {
                                vspQueues.emplace_back(std::move(spQueue));
                                ++it;
                            }
                            else
                            {
                                it = queues.erase(it);
                            }
                        }
                        return vspQueues;
                    }

                public:
                    //-----------------------------------------------------------------------------
                    DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl const &) -> DevCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl &&) -> DevCpuImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ~DevCpuImpl() = default;

                    ALPAKA_FN_HOST auto GetAllQueues() const
                    -> std::vector<std::shared_ptr<queue::cpu::ICpuQueue>>
                    {
                        return GetAllQueueImpls(m_queues);
                    }

                    //-----------------------------------------------------------------------------
                    //! Registers the given queue on this device.
                    //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                    ALPAKA_FN_HOST auto RegisterQueue(std::shared_ptr<queue::cpu::ICpuQueue> spQueue)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this queue on the device.
                        m_queues.push_back(spQueue);
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::vector<std::weak_ptr<queue::cpu::ICpuQueue>> mutable m_queues;
                };
            }
        }

        //#############################################################################
        //! The CPU device handle.
        class DevCpu : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevCpu>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfCpu>;
        protected:
            //-----------------------------------------------------------------------------
            DevCpu() :
                m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu const &) = default;
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu const &) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu &&) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator==(DevCpu const &) const
            -> bool
            {
                return true;
            }
            //-----------------------------------------------------------------------------
            auto operator!=(DevCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevCpu() = default;

        public:
            std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device name get trait specialization.
            template<>
            struct GetName<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCpu const & dev)
                -> std::string
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getTotalGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getFreeGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            template<>
            struct Reset<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    alpaka::ignore_unused(dev);

                    // The CPU does nothing on reset.
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
            class BufCpu;

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevCpu,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufCpu<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevCpu>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevCpu,
                queue::Blocking
            >
            {
                using type = queue::QueueCpuBlocking;
            };

            template<>
            struct QueueType<
                dev::DevCpu,
                queue::NonBlocking
            >
            {
                using type = queue::QueueCpuNonBlocking;
            };
        }
    }
}
