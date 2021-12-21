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
    namespace meta
    {
        //#############################################################################
        //! The trait is true if TDerived is derived from TBase but is not TBase itself.
        template<typename TBase, typename TDerived>
        using IsStrictBase = std::integral_constant<
            bool,
            std::is_base_of<TBase, TDerived>::value && !std::is_same<TBase, std::decay_t<TDerived>>::value>;
    } // namespace meta
} // namespace alpaka
