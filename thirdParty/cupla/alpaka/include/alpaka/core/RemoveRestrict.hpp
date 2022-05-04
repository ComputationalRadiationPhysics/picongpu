/* Copyright 2021 Rene Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
    //! Removes __restrict__ from a type
    template<typename T>
    struct remove_restrict
    {
        using type = T;
    };

#if BOOST_COMP_MSVC
    template<typename T>
    struct remove_restrict<T* __restrict>
    {
        using type = T*;
    };
#else
    template<typename T>
    struct remove_restrict<T* __restrict__>
    {
        using type = T*;
    };
#endif

    //! Helper to remove __restrict__ from a type
    template<typename T>
    using remove_restrict_t = typename remove_restrict<T>::type;
} // namespace alpaka
