/* Copyright 2022 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/intrinsic/IntrinsicFallback.hpp>
#    include <alpaka/intrinsic/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka::experimental
{
    //! The SYCL intrinsic.
    class IntrinsicGenericSycl : public concepts::Implements<ConceptIntrinsic, IntrinsicGenericSycl>
    {
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    template<>
    struct Popcount<experimental::IntrinsicGenericSycl>
    {
        static auto popcount(experimental::IntrinsicGenericSycl const&, std::uint32_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }

        static auto popcount(experimental::IntrinsicGenericSycl const&, std::uint64_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }
    };

    template<>
    struct Ffs<experimental::IntrinsicGenericSycl>
    {
        static auto ffs(experimental::IntrinsicGenericSycl const&, std::int32_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0) ? 0 : sycl::popcount(value ^ ~(-value));
        }

        static auto ffs(experimental::IntrinsicGenericSycl const&, std::int64_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0l) ? 0 : static_cast<std::int32_t>(sycl::popcount(value ^ ~(-value)));
        }
    };
} // namespace alpaka::trait

#endif
