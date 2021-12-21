/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dim/DimIntegralConst.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    namespace traits
    {
        //#############################################################################
        //! The arithmetic type dimension getter trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<std::is_arithmetic<T>::value>>
        {
            using type = DimInt<1u>;
        };
    } // namespace traits
} // namespace alpaka
