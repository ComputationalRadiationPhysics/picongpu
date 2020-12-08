/* Copyright 2014-2021 Axel Huebl
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

#include <string>

namespace picongpu
{
    /** 2D Phase Space Selection
     *
     * This structure stores the plot we want to create.
     * e.g. (py | x) from (momentum | spatial-coordinate)
     */
    struct AxisDescription
    {
        /** px, py or pz: \see element_momentum*/
        uint32_t momentum;
        /** x, y or z: \see element_coordinate */
        uint32_t space;

        /** short hand enums */
        enum element_momentum
        {
            px = 0u,
            py = 1u,
            pz = 2u
        };

        enum element_coordinate
        {
            x = 0u,
            y = 1u,
            z = 2u
        };

        std::string momentumAsString() const
        {
            switch(momentum)
            {
            case px:
                return "px";
            case py:
                return "py";
            case pz:
                return "pz";
            default:
                throw std::runtime_error("Unreachable!");
            }
        }

        std::string spaceAsString() const
        {
            switch(space)
            {
            case x:
                return "x";
            case y:
                return "y";
            case z:
                return "z";
            default:
                throw std::runtime_error("Unreachable!");
            }
        }
    };

} /* namespace picongpu */
