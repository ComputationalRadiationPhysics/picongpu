/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <array>
#include <cstdint>

namespace alpaka::rand::engine
{
    /** Philox backend using std::array for Key and Counter storage
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     */
    template<typename TParams>
    class PhiloxBaseStdArray
    {
    public:
        using Counter = std::array<std::uint32_t, TParams::counterSize>; ///< Counter type = std::array
        using Key = std::array<std::uint32_t, TParams::counterSize / 2>; ///< Key type = std::array
        template<typename TScalar>
        using ResultContainer
            = std::array<TScalar, TParams::counterSize>; ///< Vector template for distribution results
    };
} // namespace alpaka::rand::engine
