/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/cpu/SysInfo.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/wait/Traits.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace alpaka
{
    class DevCpu;
    namespace cpu
    {
        using ICpuQueue = IGenericThreadsQueue<DevCpu>;
    }
    namespace traits
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }
    class PltfCpu;

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
            public:
                //-----------------------------------------------------------------------------
                DevCpuImpl() = default;
                //-----------------------------------------------------------------------------
                DevCpuImpl(DevCpuImpl const&) = delete;
                //-----------------------------------------------------------------------------
                DevCpuImpl(DevCpuImpl&&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(DevCpuImpl const&) -> DevCpuImpl& = delete;
                //-----------------------------------------------------------------------------
                auto operator=(DevCpuImpl&&) -> DevCpuImpl& = delete;
                //-----------------------------------------------------------------------------
                ~DevCpuImpl() = default;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto getAllExistingQueues() const -> std::vector<std::shared_ptr<cpu::ICpuQueue>>
                {
                    std::vector<std::shared_ptr<cpu::ICpuQueue>> vspQueues;

                    std::lock_guard<std::mutex> lk(m_Mutex);
                    vspQueues.reserve(m_queues.size());

                    for(auto it = m_queues.begin(); it != m_queues.end();)
                    {
                        auto spQueue(it->lock());
                        if(spQueue)
                        {
                            vspQueues.emplace_back(std::move(spQueue));
                            ++it;
                        }
                        else
                        {
                            it = m_queues.erase(it);
                        }
                    }
                    return vspQueues;
                }

                //-----------------------------------------------------------------------------
                //! Registers the given queue on this device.
                //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<cpu::ICpuQueue> spQueue) const -> void
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    // Register this queue on the device.
                    m_queues.push_back(spQueue);
                }

            private:
                std::mutex mutable m_Mutex;
                std::vector<std::weak_ptr<cpu::ICpuQueue>> mutable m_queues;
            };
        } // namespace detail
    } // namespace cpu

    //#############################################################################
    //! The CPU device handle.
    class DevCpu
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevCpu>
        , public concepts::Implements<ConceptDev, DevCpu>
    {
        friend struct traits::GetDevByIdx<PltfCpu>;

    protected:
        //-----------------------------------------------------------------------------
        DevCpu() : m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
        {
        }

    public:
        //-----------------------------------------------------------------------------
        DevCpu(DevCpu const&) = default;
        //-----------------------------------------------------------------------------
        DevCpu(DevCpu&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevCpu const&) -> DevCpu& = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevCpu&&) -> DevCpu& = default;
        //-----------------------------------------------------------------------------
        auto operator==(DevCpu const&) const -> bool
        {
            return true;
        }
        //-----------------------------------------------------------------------------
        auto operator!=(DevCpu const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~DevCpu() = default;

        ALPAKA_FN_HOST auto getAllQueues() const -> std::vector<std::shared_ptr<cpu::ICpuQueue>>
        {
            return m_spDevCpuImpl->getAllExistingQueues();
        }

        //-----------------------------------------------------------------------------
        //! Registers the given queue on this device.
        //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<cpu::ICpuQueue> spQueue) const -> void
        {
            m_spDevCpuImpl->registerQueue(spQueue);
        }

    public:
        std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU device name get trait specialization.
        template<>
        struct GetName<DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getName(DevCpu const& dev) -> std::string
            {
                alpaka::ignore_unused(dev);

                return cpu::detail::getCpuName();
            }
        };

        //#############################################################################
        //! The CPU device available memory get trait specialization.
        template<>
        struct GetMemBytes<DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getMemBytes(DevCpu const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev);

                return cpu::detail::getTotalGlobalMemSizeBytes();
            }
        };

        //#############################################################################
        //! The CPU device free memory get trait specialization.
        template<>
        struct GetFreeMemBytes<DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevCpu const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev);

                return cpu::detail::getFreeGlobalMemSizeBytes();
            }
        };

        //#############################################################################
        //! The CPU device warp size get trait specialization.
        template<>
        struct GetWarpSize<DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getWarpSize(DevCpu const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev);

                return 1u;
            }
        };

        //#############################################################################
        //! The CPU device reset trait specialization.
        template<>
        struct Reset<DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto reset(DevCpu const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                alpaka::ignore_unused(dev);

                // The CPU does nothing on reset.
            }
        };
    } // namespace traits

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    namespace traits
    {
        //#############################################################################
        //! The CPU device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevCpu, TElem, TDim, TIdx>
        {
            using type = BufCpu<TElem, TDim, TIdx>;
        };

        //#############################################################################
        //! The CPU device platform type trait specialization.
        template<>
        struct PltfType<DevCpu>
        {
            using type = PltfCpu;
        };
    } // namespace traits
    using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
    using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;

    namespace traits
    {
        template<>
        struct QueueType<DevCpu, Blocking>
        {
            using type = QueueCpuBlocking;
        };

        template<>
        struct QueueType<DevCpu, NonBlocking>
        {
            using type = QueueCpuNonBlocking;
        };
    } // namespace traits
} // namespace alpaka
