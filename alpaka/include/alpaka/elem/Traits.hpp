/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
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
    //! The element trait.
    namespace trait
    {
        //! The element type trait.
        template<typename TView, typename TSfinae = void>
        struct ElemType;
    } // namespace trait

    //! The element type trait alias template to remove the ::type.
    template<typename TView>
    using Elem = std::remove_volatile_t<typename trait::ElemType<TView>::type>;

    // Trait specializations for unsigned integral types.
    namespace trait
    {
        //! The fundamental type elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<std::is_fundamental_v<T>>>
        {
            using type = T;
        };
    } // namespace trait
} // namespace alpaka
