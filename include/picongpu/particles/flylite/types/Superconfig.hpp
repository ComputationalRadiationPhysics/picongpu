/* Copyright 2017-2021 Axel Huebl
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

#include <pmacc/types.hpp>
#include <pmacc/math/Vector.hpp>


namespace picongpu
{
    namespace flylite
    {
        namespace types
        {
            /** Ion Superconfiguration
             *
             * This is the attribute type for an ion's screened hydrogenic
             * superconfiguration.
             *
             * See for details on screened hydrogenic levels:
             *   H.-K. Chung, S.H. Hansen, H.A. Scott.
             *   *Generalized Collisional Radiative Model Using*
             *   *Screened Hydrogenic Levels*,
             *   in Modern Methods in Collisional-Radiative Modeling of Plasmas,
             *   edited by Y. Ralchenko (Springer, 2016) pp.51-79
             *
             * @tparam T_Type the float type to use, e.g. float_64
             * @tparam T_populations the number of populations to store for each ion,
             *                       range: [0, 255]
             */
            template<typename T_Type, uint8_t T_populations>
            using Superconfig = pmacc::math::Vector<T_Type, T_populations>;
        } // namespace types
    } // namespace flylite
} // namespace picongpu
