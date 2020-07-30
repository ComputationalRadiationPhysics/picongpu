/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dim/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    namespace dim
    {
        //-----------------------------------------------------------------------------
        // N(th) dimension(s).
        template<
            std::size_t N>
        using DimInt = std::integral_constant<std::size_t, N>;
    }
}
