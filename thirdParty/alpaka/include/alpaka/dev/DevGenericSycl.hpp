/* Copyright 2024 Jan Stephan, Antonio Di Pilato, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Properties.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl;

    namespace detail
    {
        class DevGenericSyclImpl
        {
        public:
            DevGenericSyclImpl(sycl::device device, sycl::context context)
                : m_device{std::move(device)}
                , m_context{std::move(context)}
            {
            }

            // Don't call this without locking first!
            auto clean_queues() -> void
            {
                // Clean up dead queues
                auto const start = std::begin(m_queues);
                auto const old_end = std::end(m_queues);
                auto const new_end = std::remove_if(start, old_end, [](auto q_ptr) { return q_ptr.expired(); });
                m_queues.erase(new_end, old_end);
            }

            auto register_queue(std::shared_ptr<QueueGenericSyclImpl> const& queue) -> void
            {
                std::lock_guard<std::shared_mutex> lock{m_mutex};

                clean_queues();
                m_queues.emplace_back(queue);
            }

            auto register_dependency(sycl::event event) -> void
            {
                std::shared_lock<std::shared_mutex> lock{m_mutex};

                for(auto& q_ptr : m_queues)
                {
                    if(auto ptr = q_ptr.lock(); ptr != nullptr)
                        ptr->register_dependency(event);
                }
            }

            auto wait()
            {
                std::shared_lock<std::shared_mutex> lock{m_mutex};

                for(auto& q_ptr : m_queues)
                {
                    if(auto ptr = q_ptr.lock(); ptr != nullptr)
                        ptr->wait();
                }
            }

            auto get_device() const -> sycl::device
            {
                return m_device;
            }

            auto get_context() const -> sycl::context
            {
                return m_context;
            }

        private:
            sycl::device m_device;
            sycl::context m_context;
            std::vector<std::weak_ptr<QueueGenericSyclImpl>> m_queues;
            std::shared_mutex mutable m_mutex;
        };
    } // namespace detail

    //! The SYCL device handle.
    template<typename TPlatform>
    class DevGenericSycl
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevGenericSycl<TPlatform>>
        , public concepts::Implements<ConceptDev, DevGenericSycl<TPlatform>>
    {
    public:
        DevGenericSycl(sycl::device device, sycl::context context)
            : m_impl{std::make_shared<detail::DevGenericSyclImpl>(std::move(device), std::move(context))}
        {
        }

        friend auto operator==(DevGenericSycl const& lhs, DevGenericSycl const& rhs) -> bool
        {
            return (lhs.m_impl == rhs.m_impl);
        }

        friend auto operator!=(DevGenericSycl const& lhs, DevGenericSycl const& rhs) -> bool
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const -> std::pair<sycl::device, sycl::context>
        {
            return std::make_pair(m_impl->get_device(), m_impl->get_context());
        }

        std::shared_ptr<detail::DevGenericSyclImpl> m_impl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device name get trait specialization.
    template<typename TPlatform>
    struct GetName<DevGenericSycl<TPlatform>>
    {
        static auto getName(DevGenericSycl<TPlatform> const& dev) -> std::string
        {
            auto const device = dev.getNativeHandle().first;
            return device.template get_info<sycl::info::device::name>();
        }
    };

    //! The SYCL device available memory get trait specialization.
    template<typename TPlatform>
    struct GetMemBytes<DevGenericSycl<TPlatform>>
    {
        static auto getMemBytes(DevGenericSycl<TPlatform> const& dev) -> std::size_t
        {
            auto const device = dev.getNativeHandle().first;
            return device.template get_info<sycl::info::device::global_mem_size>();
        }
    };

    //! The SYCL device free memory get trait specialization.
    template<typename TPlatform>
    struct GetFreeMemBytes<DevGenericSycl<TPlatform>>
    {
        static auto getFreeMemBytes(DevGenericSycl<TPlatform> const& /* dev */) -> std::size_t
        {
            static_assert(!sizeof(TPlatform), "Querying free device memory not supported for SYCL devices.");
            return std::size_t{};
        }
    };

    //! The SYCL device warp size get trait specialization.
    template<typename TPlatform>
    struct GetWarpSizes<DevGenericSycl<TPlatform>>
    {
        static auto getWarpSizes(DevGenericSycl<TPlatform> const& dev) -> std::vector<std::size_t>
        {
            auto const device = dev.getNativeHandle().first;
            std::vector<std::size_t> warp_sizes = device.template get_info<sycl::info::device::sub_group_sizes>();
            // The CPU runtime supports a sub-group size of 64, but the SYCL implementation currently does not
            auto find64 = std::find(warp_sizes.begin(), warp_sizes.end(), 64);
            if(find64 != warp_sizes.end())
                warp_sizes.erase(find64);
            // Sort the warp sizes in decreasing order
            std::sort(warp_sizes.begin(), warp_sizes.end(), std::greater<>{});
            return warp_sizes;
        }
    };

    //! The SYCL device preferred warp size get trait specialization.
    template<typename TPlatform>
    struct GetPreferredWarpSize<DevGenericSycl<TPlatform>>
    {
        static auto getPreferredWarpSize(DevGenericSycl<TPlatform> const& dev) -> std::size_t
        {
            return GetWarpSizes<DevGenericSycl<TPlatform>>::getWarpSizes(dev).front();
        }
    };

    //! The SYCL device reset trait specialization.
    template<typename TPlatform>
    struct Reset<DevGenericSycl<TPlatform>>
    {
        static auto reset(DevGenericSycl<TPlatform> const&) -> void
        {
            static_assert(!sizeof(TPlatform), "Explicit device reset not supported for SYCL devices");
        }
    };

    //! The SYCL device native handle trait specialization.
    template<typename TPlatform>
    struct NativeHandle<DevGenericSycl<TPlatform>>
    {
        [[nodiscard]] static auto getNativeHandle(DevGenericSycl<TPlatform> const& dev)
        {
            return dev.getNativeHandle();
        }
    };

    //! The SYCL device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform>
    struct BufType<DevGenericSycl<TPlatform>, TElem, TDim, TIdx>
    {
        using type = BufGenericSycl<TElem, TDim, TIdx, TPlatform>;
    };

    //! The SYCL device platform type trait specialization.
    template<typename TPlatform>
    struct PlatformType<DevGenericSycl<TPlatform>>
    {
        using type = TPlatform;
    };

    //! The thread SYCL device wait specialization.
    template<typename TPlatform>
    struct CurrentThreadWaitFor<DevGenericSycl<TPlatform>>
    {
        static auto currentThreadWaitFor(DevGenericSycl<TPlatform> const& dev) -> void
        {
            dev.m_impl->wait();
        }
    };

    //! The SYCL blocking queue trait specialization.
    template<typename TPlatform>
    struct QueueType<DevGenericSycl<TPlatform>, Blocking>
    {
        using type = detail::QueueGenericSyclBase<DevGenericSycl<TPlatform>, true>;
    };

    //! The SYCL non-blocking queue trait specialization.
    template<typename TPlatform>
    struct QueueType<DevGenericSycl<TPlatform>, NonBlocking>
    {
        using type = detail::QueueGenericSyclBase<DevGenericSycl<TPlatform>, false>;
    };
} // namespace alpaka::trait

#endif
