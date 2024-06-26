/* Copyright 2024 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
