/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <tuple>

namespace alpaka::test
{
    //! A std::tuple holding idx types.
    using TestIdxs = std::tuple<
    // size_t is most probably identical to either std::uint64_t or std::uint32_t.
    // This would lead to duplicate tests (especially test names) which is not allowed.
    // std::size_t,
#if !defined(ALPAKA_CI)
        std::int64_t,
#endif
        std::uint64_t,
        std::int32_t
#if !defined(ALPAKA_CI)
        ,
        std::uint32_t
#endif
        // index type must be >=32bit
        >;
} // namespace alpaka::test
