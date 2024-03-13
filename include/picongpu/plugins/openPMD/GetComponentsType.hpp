/* Copyright 2021-2023 Sergei Bastrakov
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

#include <pmacc/traits/GetComponentsType.hpp>

#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** Get component type trait for bools in openPMD output
         *
         * Specializes the general trait in pmacc/traits/GetComponentsType.hpp.
         * For use with the openPMD API, both files must be included.
         *
         * The reason is that ADIOS2 backend of openPMD API currently does not support bool datasets #3732.
         * So with this specialization, PIConGPU particle attributes of type bool (e.g. radiationMask,
         * transitionRadiationMask) are treated as chars.
         *
         * This requires sizeof(bool) == sizeof(char), ::type is defined only in this case.
         */
        template<>
        struct GetComponentsType<bool>
        {
            using type = std::enable_if_t<sizeof(bool) == sizeof(char), char>;
        };

    } // namespace traits
} // namespace pmacc
