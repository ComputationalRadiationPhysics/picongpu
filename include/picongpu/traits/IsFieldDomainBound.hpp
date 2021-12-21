/* Copyright 2020-2021 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <type_traits>


namespace picongpu
{
    namespace traits
    {
        /** Whether a field is geometrically bound to the domain decomposition
         *  with respect to size, guard size, and offset
         *
         * Inherits std::true_type, std::false_type or a compatible type.
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct IsFieldDomainBound : std::true_type
        {
        };

    } // namespace traits
} // namespace picongpu
