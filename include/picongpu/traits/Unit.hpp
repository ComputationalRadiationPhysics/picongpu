/* Copyright 2013-2021 Rene Widera
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

namespace picongpu
{
    namespace traits
    {
        /** Get unit of a date that is represented by an identifier
         *
         * \tparam T_Identifier any PIConGPU identifier
         * \return \p std::vector<float_64> ::get() as static public method
         *
         * Unitless identifies, see \UnitDimension, can still be scaled by a
         * factor. If they are not scaled, implement the unit as 1.0;
         * \see unitless/speciesAttributes.unitless
         */
        template<typename T_Identifier>
        struct Unit;

    } // namespace traits

} // namespace picongpu
