/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
