/* Copyright 2026 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 *
 * The implementations of Shfl::shfl(), ShflUp::shfl_up(), ShflDown::shfl_down() and ShflXor::shfl_xor() are derived
 * from Intel DPCT.
 * Copyright (C) Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * See https://llvm.org/LICENSE.txt for license information.
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::warp
{
    //! The SYCL warp.
    template<typename TDim>
    class WarpGenericSycl : public interface::Implements<alpaka::warp::ConceptWarp, WarpGenericSycl<TDim>>
    {
    public:
        using mask_type = std::uint32_t;

        WarpGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item_warp{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_warp;
    };
} // namespace alpaka::warp

// The SYCL warp operations are implemented with the SYCL group functions (group_ballot(), the group algorithms,
// get_opportunistic_group(), select_from_group(), ...), which work only on some backends and with some oneAPI
// versions:
//   - Intel/Altera FPGAs: not supported (the FPGA backend is available only up to oneAPI 2025.0, where they do not
//     work);
//   - AMD GPUs: not supported (the Codeplay plugin, available up to oneAPI 2025.2, does not implement the SPIR-V group
//     operations they use);
//   - Intel CPUs and GPUs: supported starting from oneAPI 2025.2. Up to oneAPI 2026.1 they give wrong results when
//     some work-items have returned early from the kernel, and the code calling them is not inlined, for example
//     when compiling with -O0, -fno-inline or -fno-inline-functions: the work-items that have returned are still
//     counted as members of the group. Clang-based compilers, like icpx, define __NO_INLINE__ in these cases.
//     Define ALPAKA_SYCL_DISABLE_WARP_INLINE_CHECK to disable this check, for example if the kernels using the warp
//     operations never return early;
//   - NVIDIA GPUs: supported starting from oneAPI 2025.0.
// Using the warp operations where they are not supported fails at compile time, with one of these messages.
#    define ALPAKA_SYCL_WARP_UNSUPPORTED_FPGA "The alpaka warp operations are not supported on Intel/Altera FPGAs."
#    define ALPAKA_SYCL_WARP_UNSUPPORTED_AMD                                                                          \
        "The alpaka warp operations are not supported by the SYCL backend on AMD GPUs."
#    if ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 2, 0)
#        define ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL                                                                    \
            "The alpaka warp operations require oneAPI 2025.2 or newer on Intel CPUs and GPUs."
#    elif ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2026, 2, 0) && defined(__NO_INLINE__)                              \
        && !defined(ALPAKA_SYCL_DISABLE_WARP_INLINE_CHECK)
#        define ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL                                                                    \
            "The alpaka warp operations give wrong results on Intel CPUs and GPUs with oneAPI 2026.1 and older, "     \
            "when the SYCL code is compiled without inlining (e.g. -O0, -fno-inline or -fno-inline-functions). "      \
            "Enable optimisations (-O1 or higher) and inlining, or define ALPAKA_SYCL_DISABLE_WARP_INLINE_CHECK if "  \
            "the kernels never return early."
#    endif
#    if ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 0, 0)
#        define ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA                                                                   \
            "The alpaka warp operations require oneAPI 2025.0 or newer on NVIDIA GPUs."
#    endif

// Each SYCL kernel is compiled for all the targets, so the check is done in the device compilation for each target.
#    ifdef __SYCL_DEVICE_ONLY__
#        if defined(__NVPTX__)
#            ifdef ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA
#                define ALPAKA_SYCL_WARP_UNSUPPORTED ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA
#            endif
#        elif defined(__AMDGCN__)
#            define ALPAKA_SYCL_WARP_UNSUPPORTED ALPAKA_SYCL_WARP_UNSUPPORTED_AMD
#        elif defined(ALPAKA_SYCL_ONEAPI_FPGA) && !defined(ALPAKA_SYCL_ONEAPI_GPU)                                    \
            && !(defined(__SYCL_TARGET_INTEL_X86_64__) && __SYCL_TARGET_INTEL_X86_64__)
// The FPGA target does not define a specific macro. The SYCL headers define all the __SYCL_TARGET_*__ macros, with a
// value of 1 for the current target and 0 for the others.
#            define ALPAKA_SYCL_WARP_UNSUPPORTED ALPAKA_SYCL_WARP_UNSUPPORTED_FPGA
#        elif defined(ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL)
// Intel CPUs and GPUs, and generic SPIR-V targets
#            define ALPAKA_SYCL_WARP_UNSUPPORTED ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL
#        endif
#    endif

namespace alpaka::warp::detail
{
    //! Return why the SYCL warp operations are not supported for the accelerator tag TTag, or nullptr if they are
    //! supported.
    template<typename TTag>
    constexpr auto syclWarpUnsupportedReason() -> char const*
    {
        if constexpr(std::is_same_v<TTag, TagFpgaSyclIntel>)
        {
            return ALPAKA_SYCL_WARP_UNSUPPORTED_FPGA;
        }
        else if constexpr(std::is_same_v<TTag, TagGpuSyclAmd>)
        {
            return ALPAKA_SYCL_WARP_UNSUPPORTED_AMD;
        }
        else if constexpr(std::is_same_v<TTag, TagGpuSyclNvidia>)
        {
#    ifdef ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA
            return ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA;
#    else
            return nullptr;
#    endif
        }
        else
        {
            // Intel CPUs and GPUs, and generic SYCL targets
#    ifdef ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL
            return ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL;
#    else
            return nullptr;
#    endif
        }
    }

    // Dependent on the template parameter, so that the static_assert is only evaluated when a warp operation is used.
    template<typename TDim>
    inline constexpr bool syclWarpDependentFalse = false;
} // namespace alpaka::warp::detail

#    ifdef ALPAKA_SYCL_WARP_UNSUPPORTED
#        define ALPAKA_SYCL_WARP_CHECK                                                                                \
            static_assert(alpaka::warp::detail::syclWarpDependentFalse<TDim>, ALPAKA_SYCL_WARP_UNSUPPORTED)
#    else
#        define ALPAKA_SYCL_WARP_CHECK static_assert(true)
#    endif

namespace alpaka::warp::trait
{
    // oneAPI up to 2025.3 uses sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group(),
    // while oneAPI 2026.0 uses sycl::ext::oneapi::experimental::this_work_item::get_opportunistic_group()
#    if ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2026, 0, 0)
    using sycl::ext::oneapi::experimental::this_work_item::get_opportunistic_group;
#    else
    using sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group;
#    endif

    template<typename TDim>
    struct GetSize<warp::WarpGenericSycl<TDim>>
    {
        static auto getSize(warp::WarpGenericSycl<TDim> const& warp) -> std::int32_t
        {
            auto const sub_group = warp.m_item_warp.get_sub_group();
            // SYCL sub-groups are always 1D
            return static_cast<std::int32_t>(sub_group.get_max_local_range()[0]);
        }
    };

    template<typename TDim>
    struct GetSizeCompileTime<warp::WarpGenericSycl<TDim>>
    {
        static constexpr auto getSizeCompileTime() -> std::int32_t
        {
            // SYCL sub-groups size is usually not known at compile time
            return 0;
        }
    };

    template<typename TDim>
    struct GetSizeUpperLimit<warp::WarpGenericSycl<TDim>>
    {
        static constexpr auto getSizeUpperLimit() -> std::int32_t
        {
            // See include/alpaka/kernel/SyclSubgroupSize.hpp for possible sub-group sizes.
            return 64;
        }
    };

    template<typename TDim>
    struct Activemask<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto activemask(warp::WarpGenericSycl<TDim> const& /*warp*/) -> warp::WarpGenericSycl<TDim>::mask_type
        {
            ALPAKA_SYCL_WARP_CHECK;

            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto const mask = sycl::ext::oneapi::group_ballot(sg, true);
            std::uint32_t bits = 0;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<typename TDim>
    struct All<warp::WarpGenericSycl<TDim>>
    {
        static auto all(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate) -> std::int32_t
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto activegroup = get_opportunistic_group();
            return static_cast<std::int32_t>(sycl::all_of_group(activegroup, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Any<warp::WarpGenericSycl<TDim>>
    {
        static auto any(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate) -> std::int32_t
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto activegroup = get_opportunistic_group();
            return static_cast<std::int32_t>(sycl::any_of_group(activegroup, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Ballot<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto ballot(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate)
            -> warp::WarpGenericSycl<TDim>::mask_type
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto sub_group = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto const mask = sycl::ext::oneapi::group_ballot(sub_group, static_cast<bool>(predicate));
            // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
            // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
            // Restrict to warpSize <= 32 for now.
            std::uint32_t bits = 0;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<typename TDim>
    struct Shfl<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::int32_t srcLane,
            std::int32_t width)
        {
            ALPAKA_SYCL_WARP_CHECK;

            ALPAKA_ASSERT_ACC(width > 0);
            ALPAKA_ASSERT_ACC(srcLane >= 0);

            /* If width < srcLane the sub-group needs to be split into assumed subdivisions. The first item of each
               subdivision has the assumed index 0. The srcLane index is relative to the subdivisions.

               Example: If we assume a sub-group size of 32 and a width of 16 we will receive two subdivisions:
               The first starts at sub-group index 0 and the second at sub-group index 16. For srcLane = 4 the
               first subdivision will access the value at sub-group index 4 and the second at sub-group index 20. */
            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const start_index = actual_group.get_local_linear_id() / w * w;
            return sycl::select_from_group(actual_group, value, start_index + static_cast<std::uint32_t>(srcLane) % w);
        }
    };

    template<typename TDim>
    struct ShflUp<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_up(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::uint32_t offset, /* must be the same for all work-items in the group */
            std::int32_t width)
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const start_index = id / w * w;
            T result = sycl::shift_group_right(actual_group, value, offset);
            if((id - start_index) < offset)
            {
                result = value;
            }
            return result;
        }
    };

    template<typename TDim>
    struct ShflDown<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_down(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::uint32_t offset,
            std::int32_t width)
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const end_index = (id / w + 1) * w;
            T result = sycl::shift_group_left(actual_group, value, offset);
            if((id + offset) >= end_index)
            {
                result = value;
            }
            return result;
        }
    };

    template<typename TDim>
    struct ShflXor<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_xor(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::int32_t mask,
            std::int32_t width)
        {
            ALPAKA_SYCL_WARP_CHECK;

            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const start_index = id / w * w;
            std::uint32_t const target_offset = (id % w) ^ static_cast<std::uint32_t>(mask);
            return sycl::select_from_group(actual_group, value, target_offset < w ? start_index + target_offset : id);
        }
    };
} // namespace alpaka::warp::trait

#    undef ALPAKA_SYCL_WARP_CHECK
#    undef ALPAKA_SYCL_WARP_UNSUPPORTED
#    undef ALPAKA_SYCL_WARP_UNSUPPORTED_FPGA
#    undef ALPAKA_SYCL_WARP_UNSUPPORTED_AMD
#    undef ALPAKA_SYCL_WARP_UNSUPPORTED_INTEL
#    undef ALPAKA_SYCL_WARP_UNSUPPORTED_NVIDIA

#endif
