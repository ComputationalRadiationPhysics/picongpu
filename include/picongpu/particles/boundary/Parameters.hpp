/* Copyright 2021-2023 Lennert Sprenger, Sergei Bastrakov
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

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Basic parameters to be passed to some particle boundary condition functors
            struct Parameters
            {
                //! Axis of the active boundary
                uint32_t axis;

                /** Begin of the internal (relative to boundary) cells in total coordinates along the axis
                 *
                 * Particles with totalCellIdx[axis] < beginInternalCellsTotal are outside
                 */
                int32_t beginInternalCellsTotal;

                /** End of the internal (relative to boundary) cells in total coordinates along the axis
                 *
                 * Particles with totalCellIdx[axis] >= endInternalCellsTotal are outside
                 */
                int32_t endInternalCellsTotal;
            };
        } // namespace boundary
    } // namespace particles
} // namespace picongpu
