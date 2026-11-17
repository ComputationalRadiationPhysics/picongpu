/* Copyright 2024 Jan Stephan, Antonio Di Pilato, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dev/common/DeviceProperties.hpp"
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
    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait

    template<concepts::Tag TTag>
    using QueueGenericSyclBlocking = detail::QueueGenericSyclBase<TTag, true>;

    template<concepts::Tag TTag>
    using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TTag, false>;

    template<concepts::Tag TTag>
    struct PlatformGenericSycl;

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

            auto deviceProperties() -> std::optional<alpaka::DeviceProperties>&
            {
                std::call_once(
                    m_onceFlag,
                    [&]()
                    {
                        m_deviceProperties = std::make_optional<alpaka::DeviceProperties>();
                        auto const& device = this->get_device();
                        m_deviceProperties->name = device.template get_info<sycl::info::device::name>();
                        m_deviceProperties->totalGlobalMem
                            = device.template get_info<sycl::info::device::global_mem_size>();

                        std::vector<std::size_t> warp_sizes
                            = device.template get_info<sycl::info::device::sub_group_sizes>();
                        // The CPU runtime supports a sub-group size of 64, but the SYCL implementation currently
                        // does not. Other devices, like AMD GPUs, may support only a sub-group size of 64.
                        if(device.is_cpu())
                        {
                            auto find64 = std::find(warp_sizes.begin(), warp_sizes.end(), 64);
                            if(find64 != warp_sizes.end())
                                warp_sizes.erase(find64);
                        }
                        // Sort the warp sizes in decreasing order
                        std::sort(warp_sizes.begin(), warp_sizes.end(), std::greater<>{});
                        m_deviceProperties->warpSizes = std::move(warp_sizes);
                        m_deviceProperties->preferredWarpSize = m_deviceProperties->warpSizes.front();
                    });

                return m_deviceProperties;
            }

        private:
            sycl::device m_device;
            sycl::context m_context;
            std::vector<std::weak_ptr<QueueGenericSyclImpl>> m_queues;
            std::optional<alpaka::DeviceProperties> m_deviceProperties;
            std::shared_mutex mutable m_mutex;
            std::once_flag m_onceFlag;
        };
    } // namespace detail

    //! The SYCL device handle.
    template<concepts::Tag TTag>
    class DevGenericSycl
        : public interface::Implements<ConceptCurrentThreadWaitFor, DevGenericSycl<TTag>>
        , public interface::Implements<ConceptDev, DevGenericSycl<TTag>>
    {
        friend struct trait::GetDevByIdx<PlatformGenericSycl<TTag>>;

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

    namespace trait
    {

        //! The SYCL device name get trait specialization.
        template<concepts::Tag TTag>
        struct GetName<DevGenericSycl<TTag>>
        {
            static auto getName(DevGenericSycl<TTag> const& dev) -> std::string
            {
                return dev.m_impl->deviceProperties()->name;
            }
        };

        //! The SYCL device available memory get trait specialization.
        template<concepts::Tag TTag>
        struct GetMemBytes<DevGenericSycl<TTag>>
        {
            static auto getMemBytes(DevGenericSycl<TTag> const& dev) -> std::size_t
            {
                return dev.m_impl->deviceProperties()->totalGlobalMem;
            }
        };

        //! The SYCL device free memory get trait specialization.
        template<concepts::Tag TTag>
        struct GetFreeMemBytes<DevGenericSycl<TTag>>
        {
            static auto getFreeMemBytes(DevGenericSycl<TTag> const& /* dev */) -> std::size_t
            {
                static_assert(
                    !sizeof(PlatformGenericSycl<TTag>),
                    "Querying free device memory not supported for SYCL devices.");
                return std::size_t{};
            }
        };

        //! The SYCL device warp size get trait specialization.
        template<concepts::Tag TTag>
        struct GetWarpSizes<DevGenericSycl<TTag>>
        {
            static auto getWarpSizes(DevGenericSycl<TTag> const& dev) -> std::vector<std::size_t>
            {
                return dev.m_impl->deviceProperties()->warpSizes;
            }
        };

        //! The SYCL device preferred warp size get trait specialization.
        template<concepts::Tag TTag>
        struct GetPreferredWarpSize<DevGenericSycl<TTag>>
        {
            static auto getPreferredWarpSize(DevGenericSycl<TTag> const& dev) -> std::size_t
            {
                return dev.m_impl->deviceProperties()->preferredWarpSize;
            }
        };

        //! The SYCL device reset trait specialization.
        template<concepts::Tag TTag>
        struct Reset<DevGenericSycl<TTag>>
        {
            static auto reset(DevGenericSycl<TTag> const&) -> void
            {
                static_assert(
                    !sizeof(PlatformGenericSycl<TTag>),
                    "Explicit device reset not supported for SYCL devices");
            }
        };

        //! The SYCL device native handle trait specialization.
        template<concepts::Tag TTag>
        struct NativeHandle<DevGenericSycl<TTag>>
        {
            [[nodiscard]] static auto getNativeHandle(DevGenericSycl<TTag> const& dev)
            {
                return dev.getNativeHandle();
            }
        };

        //! The SYCL device platform type trait specialization.
        template<concepts::Tag TTag>
        struct PlatformType<DevGenericSycl<TTag>>
        {
            using type = PlatformGenericSycl<TTag>;
        };

        //! The thread SYCL device wait specialization.
        template<concepts::Tag TTag>
        struct CurrentThreadWaitFor<DevGenericSycl<TTag>>
        {
            static auto currentThreadWaitFor(DevGenericSycl<TTag> const& dev) -> void
            {
                dev.m_impl->wait();
            }
        };

        //! The SYCL blocking queue trait specialization.
        template<concepts::Tag TTag>
        struct QueueType<DevGenericSycl<TTag>, Blocking>
        {
            using type = QueueGenericSyclBlocking<TTag>;
        };

        //! The SYCL non-blocking queue trait specialization.
        template<concepts::Tag TTag>
        struct QueueType<DevGenericSycl<TTag>, NonBlocking>
        {
            using type = QueueGenericSyclNonBlocking<TTag>;
        };

    } // namespace trait
} // namespace alpaka

#endif
