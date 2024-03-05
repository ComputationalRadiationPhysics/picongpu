/* Copyright 2023 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

#include <cstddef>
#include <exception>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL device manager.
    template<typename TSelector>
    struct PlatformGenericSycl : concepts::Implements<ConceptPlatform, PlatformGenericSycl<TSelector>>
    {
        PlatformGenericSycl()
            : platform{TSelector{}}
            , devices(platform.get_devices())
            , context{sycl::context{
                  devices,
                  [](sycl::exception_list exceptions)
                  {
                      auto ss_err = std::stringstream{};
                      ss_err << "Caught asynchronous SYCL exception(s):\n";
                      for(std::exception_ptr e : exceptions)
                      {
                          try
                          {
                              std::rethrow_exception(e);
                          }
                          catch(sycl::exception const& err)
                          {
                              ss_err << err.what() << " (" << err.code() << ")\n";
                          }
                      }
                      throw std::runtime_error(ss_err.str());
                  }}}
        {
        }

        [[nodiscard]] auto syclPlatform() -> sycl::platform&
        {
            return platform;
        }

        [[nodiscard]] auto syclPlatform() const -> sycl::platform const&
        {
            return platform;
        }

        [[nodiscard]] auto syclDevices() -> std::vector<sycl::device>&
        {
            return devices;
        }

        [[nodiscard]] auto syclDevices() const -> std::vector<sycl::device> const&
        {
            return devices;
        }

        [[nodiscard]] auto syclContext() -> sycl::context&
        {
            return context;
        }

        [[nodiscard]] auto syclContext() const -> sycl::context const&
        {
            return context;
        }

    private:
        sycl::platform platform;
        std::vector<sycl::device> devices;
        sycl::context context;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL platform device count get trait specialization.
    template<typename TSelector>
    struct GetDevCount<PlatformGenericSycl<TSelector>>
    {
        static auto getDevCount(PlatformGenericSycl<TSelector> const& platform) -> std::size_t
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return platform.syclDevices().size();
        }
    };

    //! The SYCL platform device get trait specialization.
    template<typename TSelector>
    struct GetDevByIdx<alpaka::PlatformGenericSycl<TSelector>>
    {
        static auto getDevByIdx(PlatformGenericSycl<TSelector> const& platform, std::size_t const& devIdx)
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            auto const& devices = platform.syclDevices();
            if(devIdx >= devices.size())
            {
                auto ss_err = std::stringstream{};
                ss_err << "Unable to return device handle for device " << devIdx << ". There are only "
                       << devices.size() << " SYCL devices!";
                throw std::runtime_error(ss_err.str());
            }

            auto sycl_dev = devices.at(devIdx);

            // Log this device.
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            printDeviceProperties(sycl_dev);
#    elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cout << __func__ << sycl_dev.template get_info<sycl::info::device::name>() << '\n';
#    endif
            using SyclPlatform = alpaka::PlatformGenericSycl<TSelector>;
            return typename DevType<SyclPlatform>::type{sycl_dev, platform.syclContext()};
        }

    private:
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        //! Prints all the device properties to std::cout.
        static auto printDeviceProperties(sycl::device const& device) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            constexpr auto KiB = std::size_t{1024};
            constexpr auto MiB = KiB * KiB;

            std::cout << "Device type: ";
            switch(device.get_info<sycl::info::device::device_type>())
            {
            case sycl::info::device_type::cpu:
                std::cout << "CPU";
                break;

            case sycl::info::device_type::gpu:
                std::cout << "GPU";
                break;

            case sycl::info::device_type::accelerator:
                std::cout << "Accelerator";
                break;

            case sycl::info::device_type::custom:
                std::cout << "Custom";
                break;

            case sycl::info::device_type::automatic:
                std::cout << "Automatic";
                break;

            case sycl::info::device_type::host:
                std::cout << "Host";
                break;

            // The SYCL spec forbids the return of device_type::all
            // Including this here to prevent warnings because of
            // missing cases
            case sycl::info::device_type::all:
                std::cout << "All";
                break;
            }
            std::cout << '\n';

            std::cout << "Name: " << device.get_info<sycl::info::device::name>() << '\n';

            std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << '\n';

            std::cout << "Vendor ID: " << device.get_info<sycl::info::device::vendor_id>() << '\n';

            std::cout << "Driver version: " << device.get_info<sycl::info::device::driver_version>() << '\n';

            std::cout << "SYCL version: " << device.get_info<sycl::info::device::version>() << '\n';

#        if !defined(BOOST_COMP_ICPX)
            // Not defined by Level Zero back-end
            std::cout << "Backend version: " << device.get_info<sycl::info::device::backend_version>() << '\n';
#        endif

            std::cout << "Aspects: " << '\n';

#        if defined(BOOST_COMP_ICPX)
#            if BOOST_COMP_ICPX >= BOOST_VERSION_NUMBER(53, 2, 0)
            // These aspects are missing from oneAPI versions < 2023.2.0
            if(device.has(sycl::aspect::emulated))
                std::cout << "\t* emulated\n";

            if(device.has(sycl::aspect::host_debuggable))
                std::cout << "\t* debuggable using standard debuggers\n";
#            endif
#        endif

            if(device.has(sycl::aspect::fp16))
                std::cout << "\t* supports sycl::half precision\n";

            if(device.has(sycl::aspect::fp64))
                std::cout << "\t* supports double precision\n";

            if(device.has(sycl::aspect::atomic64))
                std::cout << "\t* supports 64-bit atomics\n";

            if(device.has(sycl::aspect::image))
                std::cout << "\t* supports images\n";

            if(device.has(sycl::aspect::online_compiler))
                std::cout << "\t* supports online compilation of device code\n";

            if(device.has(sycl::aspect::online_linker))
                std::cout << "\t* supports online linking of device code\n";

            if(device.has(sycl::aspect::queue_profiling))
                std::cout << "\t* supports queue profiling\n";

            if(device.has(sycl::aspect::usm_device_allocations))
                std::cout << "\t* supports explicit USM allocations\n";

            if(device.has(sycl::aspect::usm_host_allocations))
                std::cout << "\t* can access USM memory allocated by sycl::usm::alloc::host\n";

            if(device.has(sycl::aspect::usm_atomic_host_allocations))
                std::cout << "\t* can access USM memory allocated by sycl::usm::alloc::host atomically\n";

            if(device.has(sycl::aspect::usm_shared_allocations))
                std::cout << "\t* can access USM memory allocated by sycl::usm::alloc::shared\n";

            if(device.has(sycl::aspect::usm_atomic_shared_allocations))
                std::cout << "\t* can access USM memory allocated by sycl::usm::alloc::shared atomically\n";

            if(device.has(sycl::aspect::usm_system_allocations))
                std::cout << "\t* can access memory allocated by the system allocator\n";

            std::cout << "Available compute units: " << device.get_info<sycl::info::device::max_compute_units>()
                      << '\n';

            std::cout << "Maximum work item dimensions: ";
            auto dims = device.get_info<sycl::info::device::max_work_item_dimensions>();
            std::cout << dims << std::endl;

            std::cout << "Maximum number of work items:\n";
            auto const wi_1D = device.get_info<sycl::info::device::max_work_item_sizes<1>>();
            auto const wi_2D = device.get_info<sycl::info::device::max_work_item_sizes<2>>();
            auto const wi_3D = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            std::cout << "\t* 1D: (" << wi_1D.get(0) << ")\n";
            std::cout << "\t* 2D: (" << wi_2D.get(0) << ", " << wi_2D.get(1) << ")\n";
            std::cout << "\t* 3D: (" << wi_3D.get(0) << ", " << wi_3D.get(1) << ", " << wi_3D.get(2) << ")\n";

            std::cout << "Maximum number of work items per work-group: "
                      << device.get_info<sycl::info::device::max_work_group_size>() << '\n';

            std::cout << "Maximum number of sub-groups per work-group: "
                      << device.get_info<sycl::info::device::max_num_sub_groups>() << '\n';

            std::cout << "Supported sub-group sizes: ";
            auto const sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            for(auto const& sz : sg_sizes)
                std::cout << sz << ", ";
            std::cout << '\n';

            std::cout << "Preferred native vector width (char): "
                      << device.get_info<sycl::info::device::preferred_vector_width_char>() << '\n';

            std::cout << "Native ISA vector width (char): "
                      << device.get_info<sycl::info::device::native_vector_width_char>() << '\n';

            std::cout << "Preferred native vector width (short): "
                      << device.get_info<sycl::info::device::preferred_vector_width_short>() << '\n';

            std::cout << "Native ISA vector width (short): "
                      << device.get_info<sycl::info::device::native_vector_width_short>() << '\n';

            std::cout << "Preferred native vector width (int): "
                      << device.get_info<sycl::info::device::preferred_vector_width_int>() << '\n';

            std::cout << "Native ISA vector width (int): "
                      << device.get_info<sycl::info::device::native_vector_width_int>() << '\n';

            std::cout << "Preferred native vector width (long): "
                      << device.get_info<sycl::info::device::preferred_vector_width_long>() << '\n';

            std::cout << "Native ISA vector width (long): "
                      << device.get_info<sycl::info::device::native_vector_width_long>() << '\n';

            std::cout << "Preferred native vector width (float): "
                      << device.get_info<sycl::info::device::preferred_vector_width_float>() << '\n';

            std::cout << "Native ISA vector width (float): "
                      << device.get_info<sycl::info::device::native_vector_width_float>() << '\n';

            if(device.has(sycl::aspect::fp64))
            {
                std::cout << "Preferred native vector width (double): "
                          << device.get_info<sycl::info::device::preferred_vector_width_double>() << '\n';

                std::cout << "Native ISA vector width (double): "
                          << device.get_info<sycl::info::device::native_vector_width_double>() << '\n';
            }

            if(device.has(sycl::aspect::fp16))
            {
                std::cout << "Preferred native vector width (half): "
                          << device.get_info<sycl::info::device::preferred_vector_width_half>() << '\n';

                std::cout << "Native ISA vector width (half): "
                          << device.get_info<sycl::info::device::native_vector_width_half>() << '\n';
            }

            std::cout << "Maximum clock frequency: " << device.get_info<sycl::info::device::max_clock_frequency>()
                      << " MHz\n";

            std::cout << "Address space size: " << device.get_info<sycl::info::device::address_bits>() << "-bit\n";

            std::cout << "Maximum size of memory object allocation: "
                      << device.get_info<sycl::info::device::max_mem_alloc_size>() << " bytes\n";

            if(device.has(sycl::aspect::image))
            {
                std::cout << "Maximum number of simultaneous image object reads per kernel: "
                          << device.get_info<sycl::info::device::max_read_image_args>() << '\n';

                std::cout << "Maximum number of simultaneous image writes per kernel: "
                          << device.get_info<sycl::info::device::max_write_image_args>() << '\n';

                std::cout << "Maximum 1D/2D image width: " << device.get_info<sycl::info::device::image2d_max_width>()
                          << " px\n";

                std::cout << "Maximum 2D image height: " << device.get_info<sycl::info::device::image2d_max_height>()
                          << " px\n";

                std::cout << "Maximum 3D image width: " << device.get_info<sycl::info::device::image3d_max_width>()
                          << " px\n";

                std::cout << "Maximum 3D image height: " << device.get_info<sycl::info::device::image3d_max_height>()
                          << " px\n";

                std::cout << "Maximum 3D image depth: " << device.get_info<sycl::info::device::image3d_max_depth>()
                          << " px\n";

                std::cout << "Maximum number of samplers per kernel: "
                          << device.get_info<sycl::info::device::max_samplers>() << '\n';
            }

            std::cout << "Maximum kernel argument size: " << device.get_info<sycl::info::device::max_parameter_size>()
                      << " bytes\n";

            std::cout << "Memory base address alignment: "
                      << device.get_info<sycl::info::device::mem_base_addr_align>() << " bit\n";

            auto print_fp_config = [](std::string const& fp, std::vector<sycl::info::fp_config> const& conf)
            {
                std::cout << fp << " precision floating-point capabilities:\n";

                auto find_and_print = [&](sycl::info::fp_config val)
                {
                    auto it = std::find(begin(conf), end(conf), val);
                    std::cout << (it == std::end(conf) ? "No" : "Yes") << '\n';
                };

                std::cout << "\t* denorm support: ";
                find_and_print(sycl::info::fp_config::denorm);

                std::cout << "\t* INF & quiet NaN support: ";
                find_and_print(sycl::info::fp_config::inf_nan);

                std::cout << "\t* round to nearest even support: ";
                find_and_print(sycl::info::fp_config::round_to_nearest);

                std::cout << "\t* round to zero support: ";
                find_and_print(sycl::info::fp_config::round_to_zero);

                std::cout << "\t* round to infinity support: ";
                find_and_print(sycl::info::fp_config::round_to_inf);

                std::cout << "\t* IEEE754-2008 FMA support: ";
                find_and_print(sycl::info::fp_config::fma);

                std::cout << "\t* correctly rounded divide/sqrt support: ";
                find_and_print(sycl::info::fp_config::correctly_rounded_divide_sqrt);

                std::cout << "\t* software-implemented floating point operations: ";
                find_and_print(sycl::info::fp_config::soft_float);
            };

            if(device.has(sycl::aspect::fp16))
            {
                auto const fp16_conf = device.get_info<sycl::info::device::half_fp_config>();
                print_fp_config("Half", fp16_conf);
            }

            auto const fp32_conf = device.get_info<sycl::info::device::single_fp_config>();
            print_fp_config("Single", fp32_conf);

            if(device.has(sycl::aspect::fp64))
            {
                auto const fp64_conf = device.get_info<sycl::info::device::double_fp_config>();
                print_fp_config("Double", fp64_conf);
            }

            std::cout << "Global memory cache type: ";
            auto has_global_mem_cache = false;
            switch(device.get_info<sycl::info::device::global_mem_cache_type>())
            {
            case sycl::info::global_mem_cache_type::none:
                std::cout << "none";
                break;

            case sycl::info::global_mem_cache_type::read_only:
                std::cout << "read-only";
                has_global_mem_cache = true;
                break;

            case sycl::info::global_mem_cache_type::read_write:
                std::cout << "read-write";
                has_global_mem_cache = true;
                break;
            }
            std::cout << '\n';

            if(has_global_mem_cache)
            {
                std::cout << "Global memory cache line size: "
                          << device.get_info<sycl::info::device::global_mem_cache_line_size>() << " bytes\n";

                std::cout << "Global memory cache size: "
                          << device.get_info<sycl::info::device::global_mem_cache_size>() / KiB << " KiB\n";
            }

            std::cout << "Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() / MiB
                      << " MiB" << std::endl;

            std::cout << "Local memory type: ";
            auto has_local_memory = false;
            switch(device.get_info<sycl::info::device::local_mem_type>())
            {
            case sycl::info::local_mem_type::none:
                std::cout << "none";
                break;

            case sycl::info::local_mem_type::local:
                std::cout << "local";
                has_local_memory = true;
                break;

            case sycl::info::local_mem_type::global:
                std::cout << "global";
                has_local_memory = true;
                break;
            }
            std::cout << '\n';

            if(has_local_memory)
                std::cout << "Local memory size: " << device.get_info<sycl::info::device::local_mem_size>() / KiB
                          << " KiB\n";

            std::cout << "Error correction support: "
                      << (device.get_info<sycl::info::device::error_correction_support>() ? "Yes" : "No") << '\n';

            auto print_memory_orders = [](std::vector<sycl::memory_order> const& mem_orders)
            {
                for(auto const& cap : mem_orders)
                {
                    switch(cap)
                    {
                    case sycl::memory_order::relaxed:
                        std::cout << "relaxed";
                        break;

                    case sycl::memory_order::acquire:
                        std::cout << "acquire";
                        break;

                    case sycl::memory_order::release:
                        std::cout << "release";
                        break;

                    case sycl::memory_order::acq_rel:
                        std::cout << "acq_rel";
                        break;

                    case sycl::memory_order::seq_cst:
                        std::cout << "seq_cst";
                        break;
#        if defined(BOOST_COMP_ICPX)
                    // Stop icpx from complaining about its own internals.
                    case sycl::memory_order::__consume_unsupported:
                        break;
#        endif
                    }
                    std::cout << ", ";
                }
                std::cout << '\n';
            };

            std::cout << "Supported memory orderings for atomic operations: ";
            auto const mem_orders = device.get_info<sycl::info::device::atomic_memory_order_capabilities>();
            print_memory_orders(mem_orders);

#        if defined(BOOST_COMP_ICPX)
#            if BOOST_COMP_ICPX >= BOOST_VERSION_NUMBER(53, 2, 0)
            // Not implemented in oneAPI < 2023.2.0
            std::cout << "Supported memory orderings for sycl::atomic_fence: ";
            auto const fence_orders = device.get_info<sycl::info::device::atomic_fence_order_capabilities>();
            print_memory_orders(fence_orders);
#            endif
#        endif

            auto print_memory_scopes = [](std::vector<sycl::memory_scope> const& mem_scopes)
            {
                for(auto const& cap : mem_scopes)
                {
                    switch(cap)
                    {
                    case sycl::memory_scope::work_item:
                        std::cout << "work-item";
                        break;

                    case sycl::memory_scope::sub_group:
                        std::cout << "sub-group";
                        break;

                    case sycl::memory_scope::work_group:
                        std::cout << "work-group";
                        break;

                    case sycl::memory_scope::device:
                        std::cout << "device";
                        break;

                    case sycl::memory_scope::system:
                        std::cout << "system";
                        break;
                    }
                    std::cout << ", ";
                }
                std::cout << '\n';
            };

            std::cout << "Supported memory scopes for atomic operations: ";
            auto const mem_scopes = device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
            print_memory_scopes(mem_scopes);

#        if defined(BOOST_COMP_ICPX)
#            if BOOST_COMP_ICPX >= BOOST_VERSION_NUMBER(53, 2, 0)
            // Not implemented in oneAPI < 2023.2.0
            std::cout << "Supported memory scopes for sycl::atomic_fence: ";
            auto const fence_scopes = device.get_info<sycl::info::device::atomic_fence_scope_capabilities>();
            print_memory_scopes(fence_scopes);
#            endif
#        endif

            std::cout << "Device timer resolution: "
                      << device.get_info<sycl::info::device::profiling_timer_resolution>() << " ns\n";

            std::cout << "Built-in kernels: ";
            auto const builtins = device.get_info<sycl::info::device::built_in_kernel_ids>();
            for(auto const& b : builtins)
                std::cout << b.get_name() << ", ";
            std::cout << '\n';

            std::cout << "Maximum number of subdevices: ";
            auto const max_subs = device.get_info<sycl::info::device::partition_max_sub_devices>();
            std::cout << max_subs << '\n';

            if(max_subs > 1)
            {
                std::cout << "Supported partition properties: ";
                auto const part_props = device.get_info<sycl::info::device::partition_properties>();
                auto has_affinity_domains = false;
                for(auto const& prop : part_props)
                {
                    switch(prop)
                    {
                    case sycl::info::partition_property::no_partition:
                        std::cout << "no partition";
                        break;

                    case sycl::info::partition_property::partition_equally:
                        std::cout << "equally";
                        break;

                    case sycl::info::partition_property::partition_by_counts:
                        std::cout << "by counts";
                        break;

                    case sycl::info::partition_property::partition_by_affinity_domain:
                        std::cout << "by affinity domain";
                        has_affinity_domains = true;
                        break;
#        if defined(BOOST_COMP_ICPX)
                    case sycl::info::partition_property::ext_intel_partition_by_cslice:
                        std::cout << "by compute slice (Intel extension; deprecated)";
                        break;
#        endif
                    }
                    std::cout << ", ";
                }
                std::cout << '\n';

                if(has_affinity_domains)
                {
                    std::cout << "Supported partition affinity domains: ";
                    auto const aff_doms = device.get_info<sycl::info::device::partition_affinity_domains>();
                    for(auto const& dom : aff_doms)
                    {
                        switch(dom)
                        {
                        case sycl::info::partition_affinity_domain::not_applicable:
                            std::cout << "not applicable";
                            break;

                        case sycl::info::partition_affinity_domain::numa:
                            std::cout << "NUMA";
                            break;

                        case sycl::info::partition_affinity_domain::L4_cache:
                            std::cout << "L4 cache";
                            break;

                        case sycl::info::partition_affinity_domain::L3_cache:
                            std::cout << "L3 cache";
                            break;

                        case sycl::info::partition_affinity_domain::L2_cache:
                            std::cout << "L2 cache";
                            break;

                        case sycl::info::partition_affinity_domain::L1_cache:
                            std::cout << "L1 cache";
                            break;

                        case sycl::info::partition_affinity_domain::next_partitionable:
                            std::cout << "next partitionable";
                            break;
                        }
                        std::cout << ", ";
                    }
                    std::cout << '\n';
                }

                std::cout << "Current partition property: ";
                switch(device.get_info<sycl::info::device::partition_type_property>())
                {
                case sycl::info::partition_property::no_partition:
                    std::cout << "no partition";
                    break;

                case sycl::info::partition_property::partition_equally:
                    std::cout << "partitioned equally";
                    break;

                case sycl::info::partition_property::partition_by_counts:
                    std::cout << "partitioned by counts";
                    break;

                case sycl::info::partition_property::partition_by_affinity_domain:
                    std::cout << "partitioned by affinity domain";
                    break;

#        if defined(BOOST_COMP_ICPX)
                case sycl::info::partition_property::ext_intel_partition_by_cslice:
                    std::cout << "partitioned by compute slice (Intel extension; deprecated)";
                    break;
#        endif
                }
                std::cout << '\n';

                std::cout << "Current partition affinity domain: ";
                switch(device.get_info<sycl::info::device::partition_type_affinity_domain>())
                {
                case sycl::info::partition_affinity_domain::not_applicable:
                    std::cout << "not applicable";
                    break;

                case sycl::info::partition_affinity_domain::numa:
                    std::cout << "NUMA";
                    break;

                case sycl::info::partition_affinity_domain::L4_cache:
                    std::cout << "L4 cache";
                    break;

                case sycl::info::partition_affinity_domain::L3_cache:
                    std::cout << "L3 cache";
                    break;

                case sycl::info::partition_affinity_domain::L2_cache:
                    std::cout << "L2 cache";
                    break;

                case sycl::info::partition_affinity_domain::L1_cache:
                    std::cout << "L1 cache";
                    break;

                case sycl::info::partition_affinity_domain::next_partitionable:
                    std::cout << "next partitionable";
                    break;
                }
                std::cout << '\n';
            }

            std::cout.flush();
        }
#    endif
    };
} // namespace alpaka::trait

#endif
