/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdint>
#include <tuple>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //#############################################################################
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
    } // namespace test
} // namespace alpaka
