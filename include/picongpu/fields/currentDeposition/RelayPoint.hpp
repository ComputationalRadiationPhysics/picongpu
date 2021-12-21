/* Copyright 2016-2021 Rene Widera
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
    namespace currentSolver
    {
        template<bool isEven>
        struct RelayPoint
        {
            /** calculate virtual point were we split our particle trajectory
             *
             * The relay point calculation differs from the ZigZag paper version in the point
             * that the trajectory of a particle which does not leave the cell is not split.
             * The relay point for a particle which does not leave the cell is set to the
             * current position `x_2`
             *
             * If `i_1 == i_2` than the trajectory is not split.
             *
             * This function assumes that the shape in later steps is always evaluated
             * at grid integral points.
             *
             * @param i_1[out] offset to shift the coordinate system for the first
             *                 particle at position x_1
             * @param i_2[out] offset to shift the coordinate system for the second
             *                 particle at position x_2
             * @param x_1 begin position of the particle trajectory
             * @param x_2 end position of the particle trajectory
             * @return relay point for particle trajectory
             */
            DINLINE float_X operator()(int& i_1, int& i_2, const float_X x_1, const float_X x_2) const
            {
                using namespace pmacc;
                i_1 = math::floor(x_1);
                i_2 = math::floor(x_2);

                return i_1 == i_2 ? x_2 : math::max(i_1, i_2);
            }
        };

        template<>
        struct RelayPoint<false>
        {
            /** calculate virtual point were we split our particle trajectory
             *
             * @see RelayPoint< >::operator( ) description
             */
            DINLINE float_X operator()(int& i_1, int& i_2, const float_X x_1, const float_X x_2) const
            {
                i_1 = pmacc::math::float2int_rd(x_1 + float_X(0.5));
                i_2 = pmacc::math::float2int_rd(x_2 + float_X(0.5));

                return i_1 == i_2 ? x_2 : float_X(i_1 + i_2) / float_X(2.0);
            }
        };

    } // namespace currentSolver
} // namespace picongpu
