/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/atomic/Traits.hpp>

namespace alpaka
{
    //! The NoOp atomic ops.
    class AtomicNoOp
    {
    };

    namespace trait
    {
        //! The NoOp atomic operation.
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicNoOp, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicNoOp const& /* atomic */, T* const addr, T const& value) -> T
            {
                return TOp()(addr, value);
            }

            ALPAKA_FN_HOST static auto atomicOp(
                AtomicNoOp const& /* atomic */,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                return TOp()(addr, compare, value);
            }
        };
    } // namespace trait
} // namespace alpaka
