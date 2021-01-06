/* Copyright 2015-2021 Axel Huebl
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

namespace picongpu
{
    namespace traits
    {
        /* openPMD uses the powers of the 7 SI base measures to describe
         * the unit of a record
         * \see http://git.io/vROmP */
        constexpr uint32_t NUnitDimension = 7;

        // pre-C++11 "scoped enumerator" work-around
        namespace SIBaseUnits
        {
            enum SIBaseUnits_t
            {
                length = 0, // L
                mass = 1, // M
                time = 2, // T
                electricCurrent = 3, // I
                thermodynamicTemperature = 4, // theta
                amountOfSubstance = 5, // N
                luminousIntensity = 6, // J
            };
        }

    } // namespace traits
} // namespace picongpu
