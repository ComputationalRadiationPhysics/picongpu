/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The element traits.
    namespace traits
    {
        //#############################################################################
        //! The element type trait.
        template<typename TView, typename TSfinae = void>
        struct ElemType;
    } // namespace traits

    //#############################################################################
    //! The element type trait alias template to remove the ::type.
    template<typename TView>
    using Elem = std::remove_volatile_t<typename traits::ElemType<TView>::type>;

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    namespace traits
    {
        //#############################################################################
        //! The fundamental type elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<std::is_fundamental<T>::value>>
        {
            using type = T;
        };
    } // namespace traits
} // namespace alpaka
