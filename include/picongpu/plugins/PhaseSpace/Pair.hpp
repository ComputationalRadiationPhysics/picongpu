/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu::phaseSpace
{
    /** Basic implementation of std::pair
     *
     * This class is guaranteeing that the the object is trivially copyable.
     * std::pair is not giving this guarantee.
     */
    template<typename T_First, typename T_Second>
    struct Pair
    {
        T_First first;
        T_Second second;

        HDINLINE Pair(T_First inFirst, T_Second inSecond) : first{inFirst}, second{inSecond}
        {
        }

        Pair() = default;

        Pair(Pair const&) = default;
    };
} // namespace picongpu::phaseSpace
