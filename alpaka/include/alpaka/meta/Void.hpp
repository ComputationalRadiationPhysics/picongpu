/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace meta
    {
        //#############################################################################
        //! Mirror of C++17 std::void_t, maps a sequence of any types to type void
        template<class...>
        using Void = void;
    } // namespace meta
} // namespace alpaka
