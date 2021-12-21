/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    //#############################################################################
    //! The CPU fibers accelerator atomic ops.
    class AtomicNoOp
    {
    public:
        //-----------------------------------------------------------------------------
        AtomicNoOp() = default;
        //-----------------------------------------------------------------------------
        AtomicNoOp(AtomicNoOp const&) = delete;
        //-----------------------------------------------------------------------------
        AtomicNoOp(AtomicNoOp&&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicNoOp const&) -> AtomicNoOp& = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicNoOp&&) -> AtomicNoOp& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AtomicNoOp() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU fibers accelerator atomic operation.
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicNoOp, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicNoOp const& atomic, T* const addr, T const& value) -> T
            {
                alpaka::ignore_unused(atomic);
                return TOp()(addr, value);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(
                AtomicNoOp const& atomic,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                alpaka::ignore_unused(atomic);
                return TOp()(addr, compare, value);
            }
        };
    } // namespace traits
} // namespace alpaka
